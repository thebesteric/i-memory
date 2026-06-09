#!/usr/bin/env python3
"""
手动触发后台构建任务的脚本
用于测试会话构建和用户画像构建
"""
import asyncio
import datetime
import sys
import os
from contextlib import contextmanager

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pyrootutils
from dotenv import load_dotenv
from agile.utils import LogHelper

# 加载环境变量
env_path = os.path.join(pyrootutils.find_root(), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)

from src.core import user_ops
from src.core.config import env
from src.core.mem_ops import mem_ops
from src.core.db import transaction
from src.memory.memory_models import IMemoryUser
from src.memory.session.session_extractor import SessionExtractor
from src.memory.session import session_ops
from src.memory.profile.user_profile_extractor import UserProfileExtractor
from src.memory.profile import user_profile_ops
from src.memory.profile.user_profile_models import UserProfile

logger = LogHelper.get_logger(title="[MANUAL_TRIGGER]")


async def trigger_session_build(user_key: str | None = None):
    """
    手动触发会话构建（绕过时间限制，处理所有未加入会话的记忆）
    """
    logger.info("[MANUAL_TRIGGER] Starting session build...")
    
    # 查询用户
    if user_key:
        from src.memory.memory_models import IMemoryUserIdentity
        user_identity = IMemoryUserIdentity(user_key=user_key, tenant_key="test_tenant", project_key="test_project")
        user = await user_ops.get_user(user_identity)
        users = [user] if user else []
    else:
        users = await user_ops.find_user(status=1)
    
    if not users:
        logger.warning("[MANUAL_TRIGGER] No user found.")
        return
    
    success_count = 0
    for user in users:
        try:
            # 手动查询未加入会话的记忆（不限制时间）
            memories = await asyncio.to_thread(
                mem_ops.find_mem_by_conditions,
                conditions=["user_id = %s", "session_joined = 0"],
                params=[user.id],
                order_by=["created_at ASC"],
            )
            
            if not memories:
                logger.info(f"[MANUAL_TRIGGER] No un-session-joined memories for User ID: {user.id}")
                continue
            
            # 检查阈值（降低为 3 以便测试）
            threshold = 3
            if len(memories) < threshold:
                logger.info(f"[MANUAL_TRIGGER] User {user.id} has {len(memories)} memories, less than threshold {threshold}")
                continue
            
            # 自己实现会话提取和插入逻辑
            session_extractor = SessionExtractor()
            sessions = await session_extractor.invoke(memories=memories)
            
            with contextmanager(transaction)() as conn:
                sessions = await session_ops.insert_sessions(user, sessions, conn=conn)
                affected_rows = await session_ops.mark_memoires_to_session_joined([m["id"] for m in memories], conn=conn)
                logger.info(f"[MANUAL_TRIGGER] Session created for User ID: {user.id}, Associated memories: {affected_rows}")
            
            success_count += 1
            logger.info(f"[MANUAL_TRIGGER] Session build succeeded for User ID: {user.id}")
            
        except Exception as e:
            logger.error(f"[MANUAL_TRIGGER] Failed to build session for User ID: {user.id}, Error: {e}")
    
    logger.info(f"[MANUAL_TRIGGER] Session build finished, success: {success_count}/{len(users)}")


async def trigger_user_profile_build(user_key: str | None = None):
    """
    手动触发用户画像构建（绕过时间限制，处理所有未加入画像的记忆）
    """
    logger.info("[MANUAL_TRIGGER] Starting user profile build...")
    
    # 查询用户
    if user_key:
        from src.memory.memory_models import IMemoryUserIdentity
        user_identity = IMemoryUserIdentity(user_key=user_key, tenant_key="test_tenant", project_key="test_project")
        user = await user_ops.get_user(user_identity)
        users = [user] if user else []
    else:
        users = await user_ops.find_user(status=1)
    
    if not users:
        logger.warning("[MANUAL_TRIGGER] No user found.")
        return
    
    success_count = 0
    for user in users:
        try:
            # 手动查询未加入画像的记忆（不限制时间）
            memories = await asyncio.to_thread(
                mem_ops.find_mem_by_conditions,
                conditions=["user_id = %s", "profile_joined = 0"],
                params=[user.id],
                order_by=["created_at ASC"],
            )
            
            if not memories:
                logger.info(f"[MANUAL_TRIGGER] No un-profile-joined memories for User ID: {user.id}")
                continue
            
            # 检查阈值（降低为 3 以便测试）
            threshold = 3
            if len(memories) < threshold:
                logger.info(f"[MANUAL_TRIGGER] User {user.id} has {len(memories)} memories, less than threshold {threshold}")
                continue
            
            # 自己实现用户画像提取和插入逻辑
            user_profile_extractor = UserProfileExtractor()
            user_profile: UserProfile = await user_profile_extractor.invoke(user, memories=memories)
            
            with contextmanager(transaction)() as conn:
                user_profile = await user_profile_ops.upsert_user_profile(user, user_profile, conn=conn)
                affected_rows = await user_profile_ops.mark_memoires_to_profile_joined([m["id"] for m in memories], conn=conn)
                logger.info(f"[MANUAL_TRIGGER] User profile updated for User ID: {user.id}, Profile ID: {user_profile.id}, Associated memories: {affected_rows}")
            
            success_count += 1
            logger.info(f"[MANUAL_TRIGGER] User profile build succeeded for User ID: {user.id}")
            
        except Exception as e:
            logger.error(f"[MANUAL_TRIGGER] Failed to build profile for User ID: {user.id}, Error: {e}")
    
    logger.info(f"[MANUAL_TRIGGER] User profile build finished, success: {success_count}/{len(users)}")


async def main():
    """
    主函数：手动触发会话和用户画像构建
    """
    # 初始化数据库连接
    from src.core.db import get_db
    db = get_db()
    
    print("=" * 60)
    print("手动触发后台构建任务")
    print("=" * 60)
    print("1. 触发会话构建")
    print("2. 触发用户画像构建")
    print("3. 全部触发")
    print("=" * 60)
    
    choice = input("请选择 (1/2/3): ").strip()
    
    # 可选：指定用户
    user_key = input("指定用户 user_key (留空处理所有用户): ").strip() or None
    
    if choice == "1":
        await trigger_session_build(user_key)
    elif choice == "2":
        await trigger_user_profile_build(user_key)
    elif choice == "3":
        await trigger_session_build(user_key)
        print("\n" + "=" * 60 + "\n")
        await trigger_user_profile_build(user_key)
    else:
        print("无效选择")
        return
    
    print("\n" + "=" * 60)
    print("构建任务完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

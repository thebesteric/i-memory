import asyncio
from typing import Any

from src.ai.client.base_model_registrar import BaseModelRegistrar
from src.utils.log_helper import LogHelper

logger = LogHelper.get_logger()


class OpenAIRegistrar(BaseModelRegistrar):

    @staticmethod
    def is_async(client: Any) -> bool:
        return hasattr(client, "_is_async") and client._is_async

    @staticmethod
    def get_original_create(client: Any):
        return client.chat.completions.create

    async def async_wrapped_create(self, original_create, user_id, *args, **kwargs):
        # 获取历史消息
        messages = kwargs.get("messages", [])
        # 获取用户 ID
        uid = user_id or self.mem.default_user
        if messages and uid:
            try:
                # 获取最后一条用户消息
                last_msg = messages[-1]
                if last_msg.get("role") == "user":
                    # 获取用户查询内容
                    query = last_msg.get("content")
                    if isinstance(query, str):
                        # 从记忆中检索相关内容
                        context = await self.mem.search(query, user_id=uid, limit=3)
                        if context:
                            # 将上下文内容格式化
                            ctx_text = "\n".join([f"- {m['content']}" for m in context])
                            instr = f"\n\nrelevant context from memory:\n{ctx_text}"
                            # 将上下文添加到系统消息中
                            if messages[0].get("role") == "system":
                                messages[0]["content"] += instr
                            else:
                                messages.insert(0, {"role": "system", "content": instr})
                            # 更新消息列表
                            kwargs["messages"] = messages
            except Exception as e:
                logger.warning(f"failed to retrieve memory: {e}")
        # 调用原始的 create 方法
        response = await original_create(*args, **kwargs)
        try:
            # 获取用户查询内容
            query = messages[-1].get("content") if messages else ""
            # 获取模型响应内容
            answer = response.choices[0].message.content
            # 异步执行：将交互内容存储到记忆中
            asyncio.create_task(self.mem.add(f"user: {query}\nassistant: {answer}", user_id=uid))
        except Exception as e:
            logger.warning(f"failed to store interaction: {e}")

        # 返回响应结果
        return response

    def sync_wrapped_create(self, original_create, user_id, *args, **kwargs):
        # 获取历史消息
        messages = kwargs.get("messages", [])
        # 获取用户 ID
        uid = user_id or self.mem.default_user
        if messages and uid:
            try:
                # 获取最后一条用户消息
                last_msg = messages[-1]
                if last_msg.get("role") == "user":
                    # 获取用户查询内容
                    query = last_msg.get("content")
                    if isinstance(query, str):
                        try:
                            # 获取事件循环
                            loop = asyncio.get_event_loop()
                            # 如果 loop 正在运行
                            if loop.is_running():
                                # 以线程安全方式等待结果，获取上下文
                                context = asyncio.run_coroutine_threadsafe(self.mem.search(query, user_id=uid, limit=3), loop).result()
                            else:
                                # 直接运行协程，获取上下文
                                context = asyncio.run(self.mem.search(query, user_id=uid, limit=3))
                            if context:
                                # 将上下文内容格式化
                                ctx_text = "\n".join([f"- {m['content']}" for m in context])
                                instr = f"\n\nrelevant context from memory:\n{ctx_text}"
                                # 将上下文添加到系统消息中
                                if messages[0].get("role") == "system":
                                    messages[0]["content"] += instr
                                else:
                                    messages.insert(0, {"role": "system", "content": instr})
                                # 更新消息列表
                                kwargs["messages"] = messages
                        except Exception:
                            pass
            except Exception:
                pass

        # 调用原始的 create 方法
        response = original_create(*args, **kwargs)
        try:
            # 获取用户查询内容
            query = messages[-1].get("content") if messages else ""
            # 获取模型响应内容
            answer = response.choices[0].message.content
            # 运行协程，将交互内容存储到记忆中
            asyncio.run(self.mem.add(f"user: {query}\nassistant: {answer}", user_id=uid))
        except Exception:
            pass

        # 返回响应结果
        return response

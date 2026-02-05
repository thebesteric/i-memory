#!/usr/bin/env python3
"""
测试脚本：验证 ExtractEssence 类的功能
"""

import asyncio
import sys
from src.core.extract_essence import ExtractEssence

# 测试文本
TEST_CONTENT = """
今天我在上海遇见了张三，我们在2024-01-15讨论了一个重要的项目。
这个项目涉及1000万元的投资，预计需要3个月完成。
张三提出了三个建议：首先，需要加强团队合作；其次，要确保质量管理；最后，必须控制成本。
我们最终同意了这个方案，并计划下周开始执行。
这是一个很有前景的项目，相信能够取得成功。
"""

async def test_async_extract():
    """测试异步提取"""
    print("=" * 60)
    print("测试异步提取功能")
    print("=" * 60)

    extractor = ExtractEssence(
        content=TEST_CONTENT,
        sector="semantic",
        max_len=200
    )

    try:
        result = await extractor.extract()
        print(f"\n原始文本长度: {len(TEST_CONTENT)} 字符")
        print(f"提取摘要长度: {len(result)} 字符")
        print(f"\n提取结果:\n{result}")
        print("\n✓ 异步提取成功")
        return True
    except Exception as e:
        print(f"✗ 异步提取失败: {e}")
        return False

def test_sync_extract():
    """测试同步提取"""
    print("\n" + "=" * 60)
    print("测试同步提取功能")
    print("=" * 60)

    extractor = ExtractEssence(
        content=TEST_CONTENT,
        sector="event",
        max_len=200
    )

    try:
        result = extractor.extract_sync()
        print(f"\n原始文本长度: {len(TEST_CONTENT)} 字符")
        print(f"提取摘要长度: {len(result)} 字符")
        print(f"\n提取结果:\n{result}")
        print("\n✓ 同步提取成功")
        return True
    except Exception as e:
        print(f"✗ 同步提取失败: {e}")
        return False

def test_short_content():
    """测试短文本（不需要摘要）"""
    print("\n" + "=" * 60)
    print("测试短文本处理")
    print("=" * 60)

    short_text = "这是一个很短的文本"
    extractor = ExtractEssence(
        content=short_text,
        sector="semantic",
        max_len=500
    )

    try:
        result = extractor.extract_sync()
        print(f"\n原始文本: {short_text}")
        print(f"处理结果: {result}")
        assert result == short_text, "短文本应该原样返回"
        print("\n✓ 短文本处理成功")
        return True
    except Exception as e:
        print(f"✗ 短文本处理失败: {e}")
        return False

async def main():
    print("\n开始测试 ExtractEssence 类\n")

    results = []

    # 测试同步提取
    results.append(("同步提取", test_sync_extract()))

    # 测试异步提取
    results.append(("异步提取", await test_async_extract()))

    # 测试短文本
    results.append(("短文本处理", test_short_content()))

    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{test_name}: {status}")

    all_passed = all(passed for _, passed in results)
    print("\n" + ("全部测试通过！" if all_passed else "部分测试失败！"))

    return 0 if all_passed else 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n测试被中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n测试异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

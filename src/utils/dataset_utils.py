import asyncio
import os
import re
import json

from datetime import datetime
from typing import Any

from agile.utils.argparser import Argparser, Argument

from src.core.sector_classify import SectorClassifier, SECTOR_KEY_INDEX_MAPPING


def compact_inner_arrays(obj):
    """
    递归处理：仅紧凑显示字典（JSON对象）内的数组，列表（JSON数组）不处理
    :param obj: 待处理的Python对象（列表/字典/基本类型）
    :return: 处理后的对象（仅字典内的数组被标记为紧凑）
    """
    if isinstance(obj, dict):
        # 处理字典：值为列表的字段，后续序列化时紧凑显示
        processed_dict = {}
        for key, value in obj.items():
            if isinstance(value, list):
                # 关键修改：去除元素间的空格，仅保留逗号
                # 原代码：f"[{', '.join(map(str, value))}]"
                # 新代码：f"[{','.join(map(str, value))}]"（去掉, 后的空格）
                processed_dict[key] = f"[{','.join(map(str, value))}]"
            else:
                processed_dict[key] = value
        return processed_dict
    elif isinstance(obj, list):
        # 处理列表（最外层/嵌套列表）：仅递归处理内部元素，不压缩列表本身
        return [compact_inner_arrays(item) for item in obj]
    else:
        # 基本类型（字符串/数字）：直接返回
        return obj


def process_json_file(input_file, output_file, indent=2):
    """
    处理JSON文件：仅压缩对象内部的数组（元素无空格），外层列表保持缩进
    :param input_file: 输入JSON文件路径
    :param output_file: 输出JSON文件路径
    :param indent: 整体缩进数
    """
    try:
        # 1. 读取并解析JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("输入文件必须是JSON列表（最外层为数组）")

        # 2. 递归处理数据：标记需要紧凑的数组
        processed_data = compact_inner_arrays(data)

        # 3. 序列化：先按正常格式生成，再替换标记的紧凑数组
        # 第一步：生成带占位符的JSON字符串
        temp_json = json.dumps(processed_data, ensure_ascii=False, indent=indent)

        # 第二步：替换占位符（去除数组字符串的引号）
        final_json = re.sub(r'"(\[.*?\])"', r'\1', temp_json)

        # 4. 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_json)

        print(f"处理完成！")
        print(f"输入文件大小：{os.path.getsize(input_file)} 字节")
        print(f"输出文件路径：{output_file}")

    except FileNotFoundError:
        print(f"错误：文件 {input_file} 不存在")
    except json.JSONDecodeError:
        print(f"错误：{input_file} 不是有效的JSON文件")
    except ValueError as e:
        print(f"错误：{e}")
    except Exception as e:
        print(f"未知错误：{e}")


def generate_hit_labels(primary_sector: str, additional_sectors: list[str]) -> list[int]:
    """
    根据命中的模块标识生成labels列表
    :param primary_sector: 主扇区，如 "semantic"
    :param additional_sectors: 辅助扇区，如 ["episodic", "emotional"]
    :return: 长度为 5 的命中列表，如 [1, 1, 0, 1, 0]
    """
    # 初始化全 0 列表（长度等于模块总数）
    labels = [0] * len(SECTOR_KEY_INDEX_MAPPING)
    # 主标签命中
    primary_idx = SECTOR_KEY_INDEX_MAPPING[primary_sector]
    labels[primary_idx] = 1
    # 遍历附加标签命中的模块，将对应位置设为1
    for sector_key in additional_sectors:
        if sector_key in SECTOR_KEY_INDEX_MAPPING:
            idx = SECTOR_KEY_INDEX_MAPPING[sector_key]
            labels[idx] = 1
    return labels


async def calc_sectors(dataset_file_path: str) -> list[dict[str, Any]]:
    with open(dataset_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts: list[str] = []
    for item in data:
        if isinstance(item, dict):
            item_text = item.get("text")
            if isinstance(item_text, str) and item_text.strip():
                texts.append(item_text.strip())

    # 对字符串列表 texts 进行去重，且保留元素的原始顺序
    dedup_texts: list[str] = list(dict.fromkeys(texts))

    results: list[dict[str, Any]] = []
    total = len(dedup_texts)
    for idx, current_text in enumerate(dedup_texts, start=1):
        classify = await SectorClassifier(content=current_text).classify()
        result = {
            "text": current_text,
            "primary": SECTOR_KEY_INDEX_MAPPING.get(classify.primary),
            "labels": generate_hit_labels(classify.primary, classify.additional),
        }
        results.append(result)
        print(f"[{idx}/{total}] {current_text[:30]}... -> {result}")

    # 将 results 写入一个新的文件
    if results:
        date_suffix = datetime.now().strftime("%Y%m%d")
        # 生成新文件名：validation.json -> validation_20240620.json
        if "." in dataset_file_path:
            name_parts = dataset_file_path.rsplit(".", 1)
            output_file_path = f"{name_parts[0]}_{date_suffix}.{name_parts[1]}"
        else:
            output_file_path = f"{dataset_file_path}_{date_suffix}"

        # 写入 JSON 文件
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            # 压缩
            process_json_file(
                input_file=output_file_path,
                output_file=output_file_path,
                indent=2
            )

            print(f"\n处理完成！结果已写入：{output_file_path}")
        except Exception as e:
            print(f"错误：写入结果文件失败 - {str(e)}")

    return results


# 示例调用
if __name__ == "__main__":
    # # 替换为你的文件路径
    # process_json_file(
    #     "/Users/wangweijun/PycharmProjects/i-memory/assets/datasets/train.json",
    #     "/Users/wangweijun/PycharmProjects/i-memory/assets/datasets/output.json",
    #     indent=2
    # )

    parser = Argparser()
    parser.add_arg(Argument(arg_name="dataset_path", default_val="/Users/wangweijun/PycharmProjects/i-memory/assets/datasets/validation.json", required=False))

    arg = parser.get_arg("--dataset_path")
    arg.current_val = arg.current_val or arg.default_val
    asyncio.run(calc_sectors(dataset_file_path=arg.current_val))

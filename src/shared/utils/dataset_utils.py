import asyncio
import os
import re
import json

from datetime import datetime
from typing import Any, Literal

import pyrootutils
from agile.utils import LogHelper
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from services.memory.sector_classify import SectorClassifier, SECTOR_KEY_INDEX_MAPPING, SectorSentenceCreator, SectorSentenceOutput

logger = LogHelper.get_logger()


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

        logger.info(f"处理完成！")
        logger.info(f"输入文件大小：{os.path.getsize(input_file)} 字节")
        logger.info(f"输出文件路径：{output_file}")

    except FileNotFoundError:
        logger.error(f"错误：文件 {input_file} 不存在")
    except json.JSONDecodeError:
        logger.error(f"错误：{input_file} 不是有效的JSON文件")
    except ValueError as e:
        logger.error(f"错误：{e}")
    except Exception as e:
        logger.error(f"未知错误：{e}")


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


async def calc_sectors(
        dataset_file_path: str | None = None,
        sentences: list[str] | None = None,
        max_concurrency: int = 5) -> list[dict[str, Any]]:
    """
    通过语句，计算领域
    :param dataset_file_path: 文件路径，文件内容为 JSON 数组，每个元素为一个对象，对象中包含 "text" 字段
    :param sentences: 句子列表（如果没有文件路径的话，可以直接传入句子列表）
    :param max_concurrency: 并发数
    :return:
    """
    texts: list[str] = []
    if dataset_file_path:
        with open(dataset_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if isinstance(item, dict):
                item_text = item.get("text")
                if isinstance(item_text, str) and item_text.strip():
                    texts.append(item_text.strip())

    elif sentences:
        for sentence in sentences:
            texts.append(sentence.strip())

    if not texts:
        raise ValueError("没有有效的文本输入，请提供 dataset_file_path 或 sentences 参数")

    # 对字符串列表 texts 进行去重，且保留元素的原始顺序
    dedup_texts: list[str] = list(dict.fromkeys(texts))

    max_concurrency = max(1, max_concurrency)

    total = len(dedup_texts)
    semaphore = asyncio.Semaphore(max_concurrency)
    progress_lock = asyncio.Lock()
    finished = 0

    async def classify_one(order_idx: int, current_text: str) -> tuple[int, dict[str, Any]]:
        nonlocal finished
        async with semaphore:
            try:
                classify = await SectorClassifier().classify(content=current_text)
                # classify = ClassifyResult(primary="episodic", additional=["semantic"], confidence=0.5, scores={"semantic": 20, "episodic": 80})
                result = {
                    "text": current_text,
                    "primary": SECTOR_KEY_INDEX_MAPPING.get(classify.primary),
                    "labels": generate_hit_labels(classify.primary, classify.additional),
                }
            except Exception as e:
                # 单条分类失败不阻塞整个任务，使用 semantic 兜底
                logger.error(f"分类失败，使用默认标签: {current_text[:30]}... - {e}")
                result = {
                    "text": current_text,
                    "primary": SECTOR_KEY_INDEX_MAPPING.get("semantic"),
                    "labels": generate_hit_labels("semantic", []),
                }

            async with progress_lock:
                finished += 1
                logger.info(f"[{finished}/{total}] {current_text[:30]}... -> {result}")

            return order_idx, result

    tasks = [
        asyncio.create_task(classify_one(order_idx, current_text))
        for order_idx, current_text in enumerate(dedup_texts)
    ]
    indexed_results = await asyncio.gather(*tasks)

    # 按输入顺序回填结果，避免并发导致顺序变化
    ordered_results: list[dict[str, Any] | None] = [None] * total
    for order_idx, result in indexed_results:
        ordered_results[order_idx] = result

    results: list[dict[str, Any]] = [item for item in ordered_results if item is not None]

    def primary_sort_key(item: dict[str, Any]) -> int:
        primary = item.get("primary")
        # 将非法 primary 放在最后，保证分组排序稳定且安全
        if isinstance(primary, int) and 0 <= primary <= 4:
            return primary
        return 999

    # 按 primary 分组排序：0, 1, 2, 3, 4
    results = sorted(results, key=primary_sort_key)

    # 将 results 写入一个新的文件
    if results:
        date_suffix = datetime.now().strftime("%Y%m%d%H%M%S")
        # 生成新文件名：validation.json -> validation__20260313095730.json
        if dataset_file_path is not None:
            if "." in dataset_file_path:
                name_parts = dataset_file_path.rsplit(".", 1)
                output_file_path = f"{name_parts[0]}_{date_suffix}.{name_parts[1]}"
            else:
                output_file_path = f"{dataset_file_path}_{date_suffix}"
        # 生成新文件名：sentences_20260313095730.json
        else:
            output_file_path = os.path.join(pyrootutils.find_root(), "assets", "datasets", f"sentences_{date_suffix}.json")

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

            logger.info(f"\n处理完成！结果已写入：{output_file_path}，共计 {len(results)} 条记录")
        except Exception as e:
            logger.error(f"错误：写入结果文件失败 - {str(e)}")

    return results


async def create_sector_sentence(sector: Literal["episodic", "semantic", "procedural", "emotional", "reflective"],
                                 num: int,
                                 max_concurrency=1):
    ssc = SectorSentenceCreator(sector=sector, num=num)
    r: SectorSentenceOutput = await ssc.create()
    if r.sector != sector:
        raise ValueError(f"❌ 生成的句子所属领域与请求不符，预期 {sector}，但得到 {r.sector}")
    logger.info(f"✅ 已成功生成 {num} 条属于 {sector} 领域的句子")
    await calc_sectors(sentences=r.sentences, max_concurrency=max_concurrency)

async def remove_duplicates_in_file(input_file: str):
    """
    从输入文件中读取 JSON 数组，去重后写入输出文件
    :param input_file: 输入文件路径，内容为 JSON 数组
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("输入文件必须是JSON列表（最外层为数组）")

        # 使用集合去重，同时保持顺序
        seen: set[str] = set()
        first_seen_index: dict[str, int] = {}
        dedup_data = []
        duplicate_count = 0

        for idx, item in enumerate(data):
            # 用稳定序列化结果做键，支持包含 list/dict 的嵌套 JSON 结构
            item_key = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if item_key not in seen:
                seen.add(item_key)
                first_seen_index[item_key] = idx
                dedup_data.append(item)
            else:
                duplicate_count += 1
                first_idx = first_seen_index.get(item_key, -1)
                logger.info(
                    f"检测到重复数据：当前索引={idx}, 首次索引={first_idx}, 内容={json.dumps(item, ensure_ascii=False)}"
                )

        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(dedup_data, f, ensure_ascii=False, indent=2)

        # 压缩
        process_json_file(input_file=input_file, output_file=input_file, indent=2)

        # 输出去重结果
        logger.info(
            f"去重完成！输入文件路径：{input_file}, 原有数量：{len(data)}, 去重后数量：{len(dedup_data)}, 重复数量：{duplicate_count}"
        )

    except Exception as e:
        logger.error(f"错误：{e}")


async def datasets_to_english(input_file: str, max_concurrency: int = 5):
    class TranslationOutput(BaseModel):
        text: str = Field(..., description="Translated English text")

    output_parser = PydanticOutputParser(pydantic_object=TranslationOutput)
    prompt_template = PromptTemplate(
        template="""
You are a professional translator.
Translate the input text into natural, fluent English.

## Rules
1. Keep the original meaning, key facts, numbers, and named entities unchanged.
2. Do not add any explanation, only return the translated text.
3. If the input is already English, keep it as-is with light grammar polishing.

## Output Format
{format_instructions}

Input text:
{text}
""",
        input_variables=["text"],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        }
    )

    from services.memory.components import get_chat_model
    chain = prompt_template | get_chat_model() | output_parser

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入文件必须是 JSON 数组（list）")

    max_concurrency = max(1, max_concurrency)
    total = len(data)
    semaphore = asyncio.Semaphore(max_concurrency)
    progress_lock = asyncio.Lock()
    finished = 0

    async def translate_one(order_idx: int, item: Any) -> tuple[int, Any]:
        nonlocal finished

        if not isinstance(item, dict):
            async with progress_lock:
                finished += 1
                logger.warning(f"[{finished}/{total}] 非字典数据，已原样保留")
            return order_idx, item

        item_text = item.get("text")
        if not isinstance(item_text, str) or not item_text.strip():
            async with progress_lock:
                finished += 1
                logger.warning(f"[{finished}/{total}] 缺少有效 text 字段，已原样保留")
            return order_idx, item

        try:
            async with semaphore:
                output: TranslationOutput = await chain.ainvoke({"text": item_text.strip()})
            new_item = dict(item)
            new_item["text"] = output.text.strip()
            async with progress_lock:
                finished += 1
                logger.info(f"[{finished}/{total}] 翻译完成")
            return order_idx, new_item
        except Exception as e:
            # 单条翻译失败不阻塞全量任务，保留原文
            async with progress_lock:
                finished += 1
                logger.error(f"[{finished}/{total}] 翻译失败，保留原文: {e}")
            return order_idx, item

    tasks = [
        asyncio.create_task(translate_one(order_idx, item))
        for order_idx, item in enumerate(data)
    ]
    indexed_results = await asyncio.gather(*tasks)

    # 并发执行后按输入索引回填，保证输出顺序与输入一致
    ordered_results: list[Any | None] = [None] * total
    for order_idx, translated_item in indexed_results:
        ordered_results[order_idx] = translated_item
    translated_data: list[Any] = [item for item in ordered_results if item is not None]

    input_dir = os.path.dirname(input_file)
    input_name = os.path.basename(input_file)
    if "." in input_name:
        stem, ext = input_name.rsplit(".", 1)
        output_file = os.path.join(input_dir, f"{stem}_english.{ext}")
    else:
        output_file = os.path.join(input_dir, f"{input_name}_english.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)

    process_json_file(input_file=output_file, output_file=output_file, indent=2)
    logger.info(f"英文数据集已保存: {output_file}")
    return output_file



# 示例调用
if __name__ == "__main__":
    # # 替换为你的文件路径
    # process_json_file(
    #     "/Users/wangweijun/PycharmProjects/i-memory/assets/datasets/train.json",
    #     "/Users/wangweijun/PycharmProjects/i-memory/assets/datasets/output.json",
    #     indent=2
    # )

    # parser = Argparser()
    # parser.add_arg(Argument(arg_name="dataset_path", default_val="/Users/wangweijun/PycharmProjects/i-memory/assets/datasets/train.json", required=False))
    # parser.add_arg(Argument(arg_name="max_concurrency", default_val="20", required=False))
    #
    # arg = parser.get_arg("--dataset_path")
    # arg.current_val = arg.current_val or arg.default_val
    # concurrency_arg = parser.get_arg("--max_concurrency")
    # concurrency_arg.current_val = concurrency_arg.current_val or concurrency_arg.default_val
    # os.environ["IM_SECTOR_CLASSIFY_CONCURRENCY"] = str(concurrency_arg.current_val)
    # logger.info(f"使用数据集路径: {arg.current_val}")
    # logger.info(f"分类并发数: {concurrency_arg.current_val}")
    # asyncio.run(calc_sectors(dataset_file_path=arg.current_val, max_concurrency=int(concurrency_arg.current_val)))

    # 生成用例
    # asyncio.run(create_sector_sentence(sector="emotional", num=50, max_concurrency=10))

    # 去重
    # asyncio.run(remove_duplicates_in_file("/Users/wangweijun/PycharmProjects/i-memory/assets/datasets/validation.json"))
    # asyncio.run(remove_duplicates_in_file("/Users/wangweijun/PycharmProjects/i-memory/assets/datasets/train.json"))

    # 中文转英文
    # asyncio.run(datasets_to_english("/Users/wangweijun/PycharmProjects/i-memory/assets/bert/datasets/validation.json", max_concurrency=20))
    asyncio.run(datasets_to_english("/assets/bert/datasets/train.json", max_concurrency=20))

    pass

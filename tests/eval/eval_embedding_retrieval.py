import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pyrootutils

from infra.ai.embedding.providers.synthetic_embed import SyntheticEmbed
from shared.config.constants import SECTOR_RELATIONSHIPS
from services.memory import embed as memory_embed


@dataclass
class MemoryItem:
    id: str
    text: str
    sector: str


@dataclass
class QueryItem:
    id: str
    text: str
    sector: str
    gold_ids: List[str]


DEFAULT_MEMORIES: List[MemoryItem] = [
    MemoryItem("m1", "今天复盘后我很焦虑，总担心项目延期。", "emotional"),
    MemoryItem("m2", "接口重构分三步：抽象层、回归测试、灰度发布。", "procedural"),
    MemoryItem("m3", "PostgreSQL 可以通过 pgvector 做向量检索。", "semantic"),
    MemoryItem("m4", "上周五和老板评审时我被追问风险，压力很大。", "episodic"),
    MemoryItem("m5", "我意识到拖延来自对失败的过度担心。", "reflective"),
    MemoryItem("m6", "排查线上故障的 SOP：先止损，再定位，再复盘。", "procedural"),
    MemoryItem("m7", "我偏好深色主题，夜间工作眼睛更舒服。", "semantic"),
    MemoryItem("m8", "昨晚开会到很晚，回家后情绪一直低落。", "episodic"),
    MemoryItem("m9", "紧张时我会先做三分钟呼吸练习来稳定情绪。", "emotional"),
    MemoryItem("m10", "这次发布延期暴露了我沟通节奏上的问题。", "reflective"),
]

DEFAULT_QUERIES: List[QueryItem] = [
    QueryItem("q1", "我最近为什么总是焦虑", "emotional", ["m1", "m9"]),
    QueryItem("q2", "接口重构应该怎么推进", "procedural", ["m2", "m6"]),
    QueryItem("q3", "向量检索在 postgres 里怎么做", "semantic", ["m3"]),
    QueryItem("q4", "上次评审被追问风险那次经历", "episodic", ["m4"]),
    QueryItem("q5", "我拖延背后的根因是什么", "reflective", ["m5", "m10"]),
    QueryItem("q6", "怎么让自己在压力下先稳定下来", "emotional", ["m9", "m1"]),
]


class Embedder:
    name: str

    async def embed(self, text: str, sector: str) -> List[float]:
        raise NotImplementedError


class SyntheticEmbedder(Embedder):
    name = "synthetic"

    def __init__(self):
        self.model = SyntheticEmbed(dim=768)

    async def embed(self, text: str, sector: str) -> List[float]:
        return await self.model.embed(text, sector=sector)


class SectorAwareModelEmbedder(Embedder):
    name = "sector_model"

    async def embed(self, text: str, sector: str) -> List[float]:
        return await memory_embed.embed(text, sector=sector)


def cosine(v1: Sequence[float], v2: Sequence[float]) -> float:
    a = np.asarray(v1, dtype=np.float64)
    b = np.asarray(v2, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def sector_penalty(query_sector: str, memory_sector: str) -> float:
    if query_sector == memory_sector:
        return 1.0
    return SECTOR_RELATIONSHIPS.get(query_sector, {}).get(memory_sector, 0.3)


async def precompute_memory_vectors(embedder: Embedder, memories: List[MemoryItem]) -> Dict[str, List[float]]:
    tasks = [embedder.embed(m.text, m.sector) for m in memories]
    vectors = await asyncio.gather(*tasks)
    return {m.id: vec for m, vec in zip(memories, vectors)}


def compute_metrics(ranked_ids: List[List[str]], gold_ids: List[List[str]], k_values: List[int]) -> Dict[str, float]:
    total = len(ranked_ids)
    recall_hits = {k: 0 for k in k_values}
    rr_sum = 0.0

    for pred, gold in zip(ranked_ids, gold_ids):
        gold_set = set(gold)
        for k in k_values:
            topk = pred[:k]
            if any(mid in gold_set for mid in topk):
                recall_hits[k] += 1

        rr = 0.0
        for rank, mid in enumerate(pred, start=1):
            if mid in gold_set:
                rr = 1.0 / rank
                break
        rr_sum += rr

    metrics = {f"Recall@{k}": recall_hits[k] / total if total else 0.0 for k in k_values}
    metrics["MRR"] = rr_sum / total if total else 0.0
    return metrics


def parse_k_values(raw: str) -> List[int]:
    vals = []
    for x in (raw or "").split(","):
        x = x.strip()
        if not x:
            continue
        k = int(x)
        if k <= 0:
            raise ValueError("k-values must be positive integers")
        vals.append(k)
    uniq = sorted(set(vals))
    if not uniq:
        raise ValueError("k-values is empty")
    return uniq


def load_dataset(dataset_path: str | None) -> Tuple[List[MemoryItem], List[QueryItem]]:
    if not dataset_path:
        return DEFAULT_MEMORIES, DEFAULT_QUERIES

    path = Path(dataset_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    raw_memories = payload.get("memories", [])
    raw_queries = payload.get("queries", [])
    if not raw_memories or not raw_queries:
        raise ValueError("dataset json must include non-empty 'memories' and 'queries'")

    memories: List[MemoryItem] = []
    for idx, m in enumerate(raw_memories):
        try:
            memories.append(MemoryItem(id=str(m["id"]), text=str(m["text"]), sector=str(m["sector"])))
        except Exception as exc:
            raise ValueError(f"invalid memory row at index {idx}: {exc}") from exc

    memory_ids = {m.id for m in memories}

    queries: List[QueryItem] = []
    for idx, q in enumerate(raw_queries):
        try:
            gold_ids = [str(x) for x in q["gold_ids"]]
            bad_gold = [gid for gid in gold_ids if gid not in memory_ids]
            if bad_gold:
                raise ValueError(f"gold_ids not found in memories: {bad_gold}")
            queries.append(
                QueryItem(
                    id=str(q["id"]),
                    text=str(q["text"]),
                    sector=str(q["sector"]),
                    gold_ids=gold_ids,
                )
            )
        except Exception as exc:
            raise ValueError(f"invalid query row at index {idx}: {exc}") from exc

    return memories, queries


async def run_eval(embedder: Embedder,
                   memories: List[MemoryItem],
                   queries: List[QueryItem],
                   top_k: int,
                   k_values: List[int]) -> Dict[str, object]:
    t0 = time.perf_counter()
    mem_vecs = await precompute_memory_vectors(embedder, memories)

    ranked_ids: List[List[str]] = []
    per_query_rows = []

    for q in queries:
        qv = await embedder.embed(q.text, q.sector)
        scored = []
        for m in memories:
            base_sim = cosine(qv, mem_vecs[m.id])
            adjusted = base_sim * sector_penalty(q.sector, m.sector)
            scored.append((m.id, adjusted, base_sim, m.sector))

        scored.sort(key=lambda x: x[1], reverse=True)
        ranking = [mid for mid, *_ in scored[:top_k]]
        ranked_ids.append(ranking)

        hit_rank = None
        for idx, mid in enumerate(ranking, start=1):
            if mid in q.gold_ids:
                hit_rank = idx
                break

        per_query_rows.append({
            "query_id": q.id,
            "query": q.text,
            "query_sector": q.sector,
            "gold_ids": q.gold_ids,
            "top_ids": ranking,
            "first_hit_rank": hit_rank,
        })

    metrics = compute_metrics(ranked_ids, [q.gold_ids for q in queries], [k for k in k_values if k <= top_k])
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "backend": embedder.name,
        "metrics": metrics,
        "latency_ms": round(elapsed_ms, 2),
        "queries": per_query_rows,
    }


def build_embedders(names: List[str]) -> List[Embedder]:
    embedders: List[Embedder] = []
    for n in names:
        if n == "synthetic":
            embedders.append(SyntheticEmbedder())
        elif n == "sector_model":
            embedders.append(SectorAwareModelEmbedder())
        else:
            raise ValueError(f"Unknown backend: {n}")
    return embedders


def _metric_keys(results: List[Dict[str, object]]) -> List[str]:
    keys = set()
    for r in results:
        metrics_obj = r.get("metrics", {})
        if isinstance(metrics_obj, dict):
            for key in metrics_obj.keys():
                if str(key).startswith("Recall@"):
                    keys.add(str(key))
    return sorted(keys, key=lambda x: int(x.split("@")[1]) if "@" in x else 9999)


def _to_float(v: object, default: float = 0.0) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return default
    return default


def print_report(results: List[Dict[str, object]]) -> None:
    recall_keys = _metric_keys(results)
    headers = ["backend", *recall_keys, "MRR", "latency_ms"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in results:
        m = r.get("metrics", {})
        if not isinstance(m, dict):
            continue
        values = [f"{float(m.get(k, 0.0)):.3f}" for k in recall_keys]
        values.extend([
            f"{float(m.get('MRR', 0.0)):.3f}",
            f"{_to_float(r.get('latency_ms', 0.0)):.2f}",
        ])
        print("| " + " | ".join([str(r["backend"]), *values]) + " |")


def write_markdown_report(results: List[Dict[str, object]], md_path: Path) -> None:
    ok_results = [r for r in results if "metrics" in r]
    recall_keys = _metric_keys(ok_results)
    metric_headers = [*recall_keys, "MRR", "latency_ms"]
    lines = [
        "# Retrieval Eval Report",
        "",
        "## Metrics",
        "",
        "| backend | " + " | ".join(metric_headers) + " |",
        "|---|" + "|".join(["---:"] * len(metric_headers)) + "|",
    ]
    for r in ok_results:
        m = r.get("metrics", {})
        if not isinstance(m, dict):
            continue
        vals = [f"{float(m.get(k, 0.0)):.3f}" for k in recall_keys]
        vals.extend([
            f"{float(m.get('MRR', 0.0)):.3f}",
            f"{_to_float(r.get('latency_ms', 0.0)):.2f}",
        ])
        lines.append("| " + " | ".join([str(r["backend"]), *vals]) + " |")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main() -> None:
    root_path = pyrootutils.find_root()
    parser = argparse.ArgumentParser(description="Compare retrieval quality between synthetic and sector-aware model embedding.")
    parser.add_argument("--backends", nargs="+", default=["synthetic", "sector_model"], choices=["synthetic", "sector_model"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--k-values", type=str, default="1,3,5,10")
    parser.add_argument("--dataset", type=str, default=f"{root_path}/tests/eval/dataset_answer_only.json")
    parser.add_argument("--output", type=str, default=f"{root_path}/tests/eval/result/business_eval_answer_only.json")
    parser.add_argument("--output-md", type=str, default=f"{root_path}/tests/eval/result/business_eval_answer_only.md")
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    k_values = parse_k_values(args.k_values)
    memories, queries = load_dataset(args.dataset)

    embedders = build_embedders(args.backends)

    results: List[Dict[str, object]] = []
    for embedder in embedders:
        try:
            res = await run_eval(embedder, memories, queries, args.top_k, k_values)
            results.append(res)
        except Exception as exc:
            results.append({
                "backend": embedder.name,
                "error": str(exc),
            })

    ok_results = [r for r in results if "metrics" in r]
    if ok_results:
        print_report(ok_results)

    err_results = [r for r in results if "error" in r]
    if err_results:
        print("\nErrors:")
        for e in err_results:
            print(f"- {e['backend']}: {e['error']}")

    output_payload = {
        "results": results,
        "dataset": {
            "memories": [m.__dict__ for m in memories],
            "queries": [q.__dict__ for q in queries],
        },
        "k_values": k_values,
        "top_k": args.top_k,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved report: {out_path}")

    if args.output_md:
        md_path = Path(args.output_md)
        write_markdown_report(results, md_path)
        print(f"Saved markdown: {md_path}")


if __name__ == "__main__":
    asyncio.run(main())


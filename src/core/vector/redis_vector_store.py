from typing import List, Optional

import numpy as np
import redis
from utils.log_helper import LogHelper
from utils.singleton import singleton

from src.core.vector.base_vector_store import BaseVectorStore, VectorRow, VectorSearch
from src.memory.models.memory_models import IMemoryFilters, IMemoryUserIdentity


logger = LogHelper.get_logger()


@singleton
class RedisVectorStore(BaseVectorStore):

    def __init__(self, url: str, prefix: str = "im:vec:"):
        self.url = url
        self.prefix = prefix
        self.client = None

    async def _get_client(self):
        if not self.client:
            self.client = redis.from_url(self.url)
        return self.client

    def _key(self, id: str) -> str:
        return f"{self.prefix}{id}"

    async def store_vector(self, id: str, sector: str, vector: List[float], dim: int, user_identity: IMemoryUserIdentity = None):
        """
        存储向量
        :param id: 唯一标识
        :param sector: 扇区名称
        :param vector: 向量列表
        :param dim: 向量维度
        :param user_identity: 用户身份
        :return:
        """
        client = await self._get_client()
        key = self._key(id)
        vec_bytes = np.array(vector, dtype=np.float32).tobytes()

        user_id = user_identity.user_id if (user_identity and user_identity.user_id) else None
        tenant_id = user_identity.tenant_id if (user_identity and user_identity.tenant_id) else None
        project_id = user_identity.project_id if (user_identity and user_identity.project_id) else None

        # 批量获取 Redis 中原有值
        original_vals = await client.hmget(key, "user_id", "tenant_id", "project_id")
        original_user_id = original_vals[0] or ""
        original_tenant_id = original_vals[1] or ""
        original_project_id = original_vals[2] or ""

        final_user_id = user_id if user_id else original_user_id
        final_tenant_id = tenant_id if tenant_id else original_tenant_id
        final_project_id = project_id if project_id else original_project_id

        mapping = {
            "id": id,
            "sector": sector,
            "dim": dim,
            "v": vec_bytes,
            "user_id": final_user_id,
            "tenant_id": final_tenant_id,
            "project_id": final_project_id
        }
        await client.hset(key, mapping=mapping)

    async def get_vectors_by_id(self, id: str) -> List[VectorRow]:
        """
        根据 ID 获取所有相关向量
        :param id: 唯一标识
        :return:
        """
        client = await self._get_client()
        key = self._key(id)
        data = await client.hgetall(key)
        if not data: return []

        def dec(x): return x.decode('utf-8') if isinstance(x, bytes) else str(x)

        vec_bytes = data.get(b'v') or data.get('v')
        vec = list(np.frombuffer(vec_bytes, dtype=np.float32))

        return [VectorRow(
            dec(data.get(b'id') or data.get('id')),
            dec(data.get(b'sector') or data.get('sector')),
            vec,
            int(dec(data.get(b'dim') or data.get('dim')))
        )]

    async def get_vector(self, id: str, sector: str) -> Optional[VectorRow]:
        """
        根据 ID 和 sector 获取单个向量
        :param id: 唯一标识
        :param sector: 扇区名称
        :return:
        """
        rows = await self.get_vectors_by_id(id)
        for r in rows:
            if r.sector == sector:
                return r
        return None

    async def delete_vectors(self, id: str):
        """
        删除指定 ID 的所有向量
        :param id: 唯一标识
        :return:
        """
        client = await self._get_client()
        await client.delete(self._key(id))

    async def search(self, vector: List[float], sector: str, k: int, filters: IMemoryFilters = None) -> List[VectorSearch]:
        """
        相似度搜索
        :param vector: 向量
        :param sector: 扇区
        :param k: 返回的相似向量数量
        :param filters: 过滤条件
        :return: 相似向量列表
        """
        client = await self._get_client()
        query_vec = np.array(vector, dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)

        cursor = 0
        results = []

        while True:
            cursor, keys = await client.scan(cursor, match=f"{self.prefix}*", count=100)
            if keys:
                pipe = client.pipeline()
                for key in keys:
                    pipe.hgetall(key)
                items = await pipe.execute()

                for item in items:
                    if not item: continue

                    def decode(x):
                        return x.decode('utf-8') if isinstance(x, bytes) else str(x)

                    # 过滤 sector
                    i_sector = decode(item.get(b'sector') or item.get('sector'))
                    if i_sector != sector:
                        continue

                    # 过滤用户身份
                    if filters and filters.user_identity:
                        user_identity = filters.user_identity
                        i_uid = decode(item.get(b'user_id') or item.get('user_id'))
                        i_tid = decode(item.get(b'tenant_id') or item.get('tenant_id'))
                        i_pid = decode(item.get(b'project_id') or item.get('project_id'))
                        if i_uid != user_identity.user_id or i_tid != user_identity.tenant_id or i_pid != user_identity.project_id:
                            continue

                    v_bytes = item.get(b'v') or item.get('v')
                    v = np.frombuffer(v_bytes, dtype=np.float32)

                    dot = np.dot(query_vec, v)
                    norm = np.linalg.norm(v)
                    sim = dot / (q_norm * norm) if (q_norm * norm) > 0 else 0

                    results.append(VectorSearch(id=decode(item.get(b'id') or item.get('id')), similarity=float(sim)))

            if cursor == 0: break

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:k]

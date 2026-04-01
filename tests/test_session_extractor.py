import asyncio
import unittest

from src.core.mem_ops import mem_ops
from src.memory.session.session_extractor import SessionExtractor
from src.memory.session.session_models import Sessions


class TestSessionExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.memories = mem_ops.find_mem_by_conditions(
            conditions=["user_id = %s", "created_at < %s"],
            params=["76d09a00-c869-4bf5-a686-b344ba3d3d3b", "2026-03-31 23:59:59"],
            order_by=["created_at ASC"],
        )

    def test(self):
        sessions: Sessions = asyncio.run(SessionExtractor().invoke(
            memories=self.memories,
        ))
        print(sessions.model_dump())
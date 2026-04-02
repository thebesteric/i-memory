import asyncio
import unittest

from src.memory.graph.graph_builder import graph_build, process_user_queue


class TestGraphBuild(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test(self):
        asyncio.run(graph_build())
        asyncio.run(process_user_queue())
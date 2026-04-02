import asyncio
import unittest

from src.memory.session.session_builder import session_build


class TestSessionExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test(self):
        asyncio.run(session_build())
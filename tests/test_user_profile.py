import asyncio
import unittest

from src.core.mem_ops import mem_ops
from src.memory.profile.user_profile_builder import describe_user_profile


class TestSessionExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test(self):
        asyncio.run(describe_user_profile())
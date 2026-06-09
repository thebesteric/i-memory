import asyncio
import unittest

from src.memory.profile.user_profile_builder import describe_user_profile


class TestDescribeUserProfile(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_describe_user_profile(self):
        asyncio.run(describe_user_profile())
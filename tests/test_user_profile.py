import asyncio
import unittest

from services.profile.user_profile_builder import describe_user_profile


class TestSessionExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test(self):
        asyncio.run(describe_user_profile())
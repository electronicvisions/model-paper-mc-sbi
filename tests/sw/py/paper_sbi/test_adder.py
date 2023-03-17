import unittest

from paper_sbi import add


class TestAdder(unittest.TestCase):
    def test_commutative(self):
        self.assertEqual(add(3, 4), add(4, 3))


if __name__ == "__main__":
    unittest.main()

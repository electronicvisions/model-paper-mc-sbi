import unittest

from paper_sbi.scripts.minimal_experiment import main


class TestMinimalExperiment(unittest.TestCase):
    def test_experiment(self):
        self.assertIsNone(main())


if __name__ == "__main__":
    unittest.main()

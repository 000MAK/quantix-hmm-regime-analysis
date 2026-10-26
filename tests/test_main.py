import unittest
from main import regime_counts

class RegimeCountsTests(unittest.TestCase):
    def test_counts_are_sorted_and_complete(self): self.assertEqual(regime_counts([2, 1, 2, 0, 1, 2]), {0: 1, 1: 2, 2: 3})
    def test_empty_sequence(self): self.assertEqual(regime_counts([]), {})

if __name__ == "__main__": unittest.main()

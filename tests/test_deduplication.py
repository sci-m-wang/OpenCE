import unittest
from ace.deduplication import Deduplicator

class DeduplicationTest(unittest.TestCase):
    def test_find_duplicates(self):
        deduplicator = Deduplicator(similarity_threshold=0.99)
        new_bullets = {
            "1": "This is a test sentence.",
            "2": "This is another test sentence.",
            "3": "This is a completely different sentence.",
        }
        existing_bullets = {
            "4": "This is a test sentence.",
            "5": "This is a third test sentence.",
        }
        duplicate_ids = deduplicator.find_duplicates(new_bullets, existing_bullets)
        self.assertEqual(sorted(duplicate_ids), ["1"])

if __name__ == "__main__":
    unittest.main()

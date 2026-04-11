import argparse
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from musubi_tuner.utils.argparse_utils import parse_bool, str2bool


class TestArgparseUtils(unittest.TestCase):
    def test_parse_bool_accepts_explicit_literals(self):
        self.assertTrue(parse_bool(True))
        self.assertTrue(parse_bool("true"))
        self.assertTrue(parse_bool(" YES "))
        self.assertTrue(parse_bool(1))
        self.assertFalse(parse_bool(None))
        self.assertFalse(parse_bool("false"))
        self.assertFalse(parse_bool("0"))
        self.assertFalse(parse_bool(""))
        self.assertFalse(parse_bool(0))

    def test_parse_bool_rejects_ambiguous_values(self):
        with self.assertRaises(ValueError):
            parse_bool("maybe")
        with self.assertRaises(ValueError):
            parse_bool(2)
        with self.assertRaises(ValueError):
            parse_bool([])

    def test_str2bool_wraps_errors_for_argparse(self):
        self.assertTrue(str2bool("yes"))
        self.assertFalse(str2bool("no"))
        with self.assertRaises(argparse.ArgumentTypeError):
            str2bool("maybe")


if __name__ == "__main__":
    unittest.main()

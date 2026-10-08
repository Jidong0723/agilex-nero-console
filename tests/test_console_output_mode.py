"""Include the dependency-free console interaction suite in full regression."""
import shutil
import subprocess
import unittest
from pathlib import Path


class ConsoleOutputModeTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for console interaction tests")
    def test_console_interactions_with_fake_dom_and_http(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([shutil.which("node"), "--test", "tests/console_output_mode.test.js"],
            cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

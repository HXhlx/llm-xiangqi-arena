"""Keep .gitignore to a known set of generic directory entries."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory lines (trailing slash) that this repo is allowed to ignore.
# A new entry has to be added here on purpose. Tool-specific private names
# are not part of this list.
_ALLOWED_DIR_IGNORES = frozenset(
    {
        "__pycache__/",
        "*.egg-info/",
        "dist/",
        "build/",
        "secrets/",
        ".secrets/",
        "cookies/",
        ".litellm_cache/",
        ".venv/",
        "venv/",
        "env/",
        ".idea/",
        ".vscode/",
        "logs/",
        "log_bak/",
        ".claude/",
        ".agents/",
        "archive/",
    }
)

_REQUIRED_GENERIC_TOOL_DIRS = frozenset({".claude/", ".agents/", "archive/"})


def _directory_ignore_lines(text: str) -> set[str]:
    found: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        if line.endswith("/"):
            found.add(line)
    return found


class GitignoreHygieneTests(unittest.TestCase):
    def test_directory_ignores_stay_on_the_generic_allowlist(self):
        text = (Path(PROJECT_ROOT) / ".gitignore").read_text(encoding="utf-8")
        dirs = _directory_ignore_lines(text)
        self.assertTrue(_REQUIRED_GENERIC_TOOL_DIRS <= dirs)
        extra = dirs - _ALLOWED_DIR_IGNORES
        self.assertEqual(extra, set())


if __name__ == "__main__":
    unittest.main()

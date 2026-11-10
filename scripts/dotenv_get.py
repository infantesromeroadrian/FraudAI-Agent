#!/usr/bin/env python3
"""Print the value of one key of a dotenv file (nothing when absent). Usage: dotenv_get.py FILE KEY."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv_util import read_dotenv

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: dotenv_get.py FILE KEY")
    sys.stdout.write(read_dotenv(Path(sys.argv[1]), {sys.argv[2]}).get(sys.argv[2], ""))

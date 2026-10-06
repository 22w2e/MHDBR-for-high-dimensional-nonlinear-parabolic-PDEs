#!/usr/bin/env python3
"""Run Example 1 with MHDBR J=1 (terminal condition only)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.example1.problem import main


if __name__ == "__main__":
    main(1)

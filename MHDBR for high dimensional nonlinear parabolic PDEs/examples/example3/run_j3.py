#!/usr/bin/env python3
"""Run Example 3 with MHDBR J=3 and refined Euler startup."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.example3.problem import main


if __name__ == "__main__":
    main(3)

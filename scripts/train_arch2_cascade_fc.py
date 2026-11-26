#!/usr/bin/env python
"""Architecture 2: 3 MP layers, 2 shared FC, cascade family → reagent → site.

Seeded 80% train / 10% val / 10% test split (parent-grouped, ``--seed`` default 42).
Writes ``output/arch2/planner.pt``.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.training.cv_train import main

if __name__ == "__main__":
    raise SystemExit(main(shared_fc=2, conditioning="cascade", out=ROOT / "output" / "arch2"))

"""Source-tree paths shared by the command-line tools."""

from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
THIRD_PARTY_ROOT = PROJECT / "third_party"

for dependency in ("taming-transformers", "clip"):
    checkout = THIRD_PARTY_ROOT / dependency
    if checkout.is_dir() and str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amber.datasets.build import build_dataset

if __name__ == "__main__":
    build_dataset()

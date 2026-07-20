from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amber.pipeline.scanner_app import main

if __name__ == "__main__":
    main(loop="--loop" in sys.argv)

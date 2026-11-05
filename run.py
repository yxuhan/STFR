"""Compatibility entrypoint; execution and logging are shared by all stages."""
from pathlib import Path
import sys

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pipeline import main
    raise SystemExit(main())
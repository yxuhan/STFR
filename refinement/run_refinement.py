"""Independent stage CLI; --plan_json is the root orchestrator protocol."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_runtime import stage_main
from workflow import build_steps

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root', required=True)
    raise SystemExit(stage_main(parser, build_steps, 'refine', Path(__file__).resolve().parents[1]))

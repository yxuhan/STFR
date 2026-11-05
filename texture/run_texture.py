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
    parser.add_argument('--fill_eye_holes', dest='tex_fill_eye_holes', action='store_true')
    parser.add_argument('--to_metahuman', type=int, choices=[0, 1], default=0)
    raise SystemExit(stage_main(parser, build_steps, 'tex', Path(__file__).resolve().parents[1]))

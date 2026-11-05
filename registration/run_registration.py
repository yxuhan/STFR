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
    parser.add_argument('--close_eye', dest='reg_close_eye', type=int, choices=[0, 1], default=0)
    parser.add_argument('--ldm_type', dest='reg_ldm_type', choices=['eyelid','ibug','interp'], default='eyelid')
    raise SystemExit(stage_main(parser, build_steps, 'reg', Path(__file__).resolve().parents[1]))

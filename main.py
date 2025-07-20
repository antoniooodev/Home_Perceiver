#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# project root on PYTHONPATH
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Run Mode A (tight outline) or Mode B (multi-class detection)"
    )
    parser.add_argument(
        "--mode", choices=["A", "B"], required=True, help="Choose Mode A or Mode B"
    )
    args = parser.parse_args()

    if args.mode == "A":
        from scripts.demo_modeA import main as modeA_main

        modeA_main()  # nessun parametro source
    else:
        from scripts.demo_modeB import main as modeB_main

        modeB_main()  # idem


if __name__ == "__main__":
    main()

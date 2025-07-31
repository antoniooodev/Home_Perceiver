#!/usr/bin/env python
"""
Build a PDF report for one run_id.

Produces:
  output/reports/<run_id>.pdf
Requires:
  output/csv/<run_id>.summary.csv
  output/json/<run_id>.classes.png
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def compile_pdf(tmp: Path) -> None:
    """Run pdflatex twice for clean refs."""
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "report.tex"],
            cwd=str(tmp),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--title", default="Visual-Detection Report")
    args = ap.parse_args()

    root = Path.cwd()
    rid  = args.run_id
    paths = {
        "csv":  root / f"output/csv/{rid}.summary.csv",
        "png":  root / f"output/json/{rid}.classes.png",
        "tpl":  root / "tools/template.tex",
        "out":  root / f"output/reports/{rid}.pdf",
    }
    if not paths["csv"].exists() or not paths["png"].exists():
        sys.exit(f"Missing artefacts for run {rid}")

    paths["out"].parent.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        shutil.copy(paths["csv"], tmp / "summary.csv")
        shutil.copy(paths["png"], tmp / "classes.png")
        shutil.copy(paths["tpl"], tmp / "report.tex")

        # ---------- table to table.tex ----------
        df = pd.read_csv(tmp / "summary.csv")
        tex_table = (
            df.to_latex(index=False, column_format="lr")
              .replace("_", r"\_")              # escape underscores
        )
        (tmp / "table.tex").write_text(tex_table)

        # ---------- patch the template ----------
        tex = (tmp / "report.tex").read_text()
        tex = tex.replace("Final Report", args.title)
        (tmp / "report.tex").write_text(tex)

        # ---------- compile ----------
        try:
            compile_pdf(tmp)
        except subprocess.CalledProcessError:
            log = (tmp / "report.log").read_text() if (tmp / "report.log").exists() else ""
            print("LaTeX failed — tail of log:\n", log[-2000:])
            sys.exit(1)

        shutil.move(tmp / "report.pdf", paths["out"])
        print("Generated:", paths["out"].relative_to(root))


if __name__ == "__main__":
    main()
from __future__ import annotations
import pathlib
import runpy


def main() -> None:
    script = pathlib.Path(__file__).resolve().parent.parent.parent / "scripts" / "run_ABCDEOOH_experiment.py"
    runpy.run_path(str(script), run_name="__main__")

"""Convenient repository entry point; always executes the project's virtualenv."""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
SOURCE = ROOT / "src/experiments"


def main():
    parser = argparse.ArgumentParser(
        description="SOC-RAG Guard: status, tests, final predictions and experiments"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    commands.add_parser("test")
    predict = commands.add_parser("predict")
    predict.add_argument("--input", required=True, type=Path)
    predict.add_argument("--output", required=True, type=Path)
    predict.add_argument(
        "--bundle",
        type=Path,
        default=ROOT / "results/research_evaluation_v1/final_model",
    )
    run = commands.add_parser("run")
    run.add_argument("script", choices=sorted(p.stem for p in SOURCE.glob("*.py")))
    run.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "status":
        selected = ROOT / "results/research_evaluation_v1/model_selection.json"
        print(
            json.dumps(
                {
                    "project": str(ROOT),
                    "python": str(PYTHON),
                    "python_exists": PYTHON.exists(),
                    "source": str(SOURCE),
                    "report": str(ROOT / "docs/reports/EXPERIMENT_REPORT.md"),
                    "final_model": (
                        json.loads(selected.read_text())["selected"]
                        if selected.exists()
                        else None
                    ),
                },
                indent=2,
            )
        )
        return
    if not PYTHON.exists():
        raise FileNotFoundError("Project virtual environment missing: " + str(PYTHON))
    if args.command == "test":
        command = [
            str(PYTHON),
            "-m",
            "unittest",
            "discover",
            "-s",
            str(ROOT / "tests"),
            "-p",
            "test_*.py",
        ]
    elif args.command == "predict":
        command = [
            str(PYTHON),
            str(SOURCE / "predict_final.py"),
            "--bundle",
            str(args.bundle),
            "--input",
            str(args.input),
            "--output",
            str(args.output),
        ]
    else:
        command = [str(PYTHON), str(SOURCE / (args.script + ".py")), *args.arguments]
    raise SystemExit(subprocess.call(command, cwd=ROOT))


if __name__ == "__main__":
    main()

import argparse
import subprocess
import sys
from pathlib import Path


def run(filename=None):
    app_path = Path(__file__).parent / "quiz_editor.py"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
    ]

    if filename:
        cmd.extend([filename])

    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Launch the quiz editor Streamlit app."
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default=None,
        help="Optional YAML file to load",
    )

    args = parser.parse_args()

    run(args.filename)


if __name__ == "__main__":
    main()
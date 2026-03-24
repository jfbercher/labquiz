import subprocess
import sys
from pathlib import Path

def main():
    app_path = Path(__file__).parent / "quiz_editor.py"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
        ]
    )
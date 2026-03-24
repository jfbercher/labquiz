import argparse
from pathlib import Path

try:
    import tomlkit  # pour préserver le format
    import tomllib  # Python ≥3.11
except ImportError:
    import tomli as tomllib
    #raise SystemExit("Installez tomlkit : pip install tomlkit")

parser = argparse.ArgumentParser()
parser.add_argument("version", help="Nouvelle version (ex: 0.3.9)")
args = parser.parse_args()

path = Path("pyproject.toml")

data = tomlkit.parse(path.read_text())
data["project"]["version"] = args.version

path.write_text(tomlkit.dumps(data))

print(f"Version mise à jour → {args.version}")

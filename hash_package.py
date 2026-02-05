import hashlib
from pathlib import Path
import argparse
from datetime import datetime
import tomlkit
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Python < 3.8
    from importlib_metadata import version, PackageNotFoundError


PACKAGE_DIR=Path('/Users/bercherj/JFB/Ens/ASSL/TP ASSL/prepa_quizzes/LabQuizPkg/src/labquiz')
PACKAGE_NAME = "labquiz"  # nom du package tel qu'il sera installé
EXCLUDE =  {"putils.py", "__pycache__", ".ipynb_checkpoints", ".DS_Store"}

def file_hash(path: Path, algo="sha256") -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def package_hash(exclude=None, algo="sha256"):
    h = hashlib.new(algo)
    exclude = exclude or set()

    files = sorted(
        p for p in PACKAGE_DIR.rglob("*")
        if p.is_file() and p.name not in exclude and  not any(ex in p.parts for ex in exclude)
    )
    print("files in package_hash", files)

    for path in files:
        h.update(path.read_bytes())

    return h.hexdigest(), files


def get_version():
    import tomllib
    from pathlib import Path
    pyproject = Path("pyproject.toml") #Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    return data["project"]["version"]
    """try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return "0.0.0"
    """


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output",
        default="hashes.txt",
        help="Fichier de sortie"
    )
    parser.add_argument(
        "--no-stdout",
        action="store_true",
        help="Ne rien afficher à l'écran"
    )
    args = parser.parse_args()

    ver = get_version()


    lines = []
    lines.append(f"# Hashes générés le {datetime.now().isoformat()}")
    lines.append(f"# Dossier analysé : {PACKAGE_DIR}")
    lines.append(f"# Version du package : {ver}")
    lines.append("")

    lines.append("[files]")
    for f in sorted(PACKAGE_DIR.rglob("*.py")):
        lines.append(f"{file_hash(f)}  {f}")

    lines.append("")
    full_hash, _ = package_hash()
    lines.append("[package]")
    lines.append(full_hash)

    lines.append("")
    partial_hash, _ = package_hash(EXCLUDE)
    lines.append("[package_without_putils]")
    lines.append(partial_hash)

    content = "\n".join(lines)
    Path(args.output).write_text(content)

    if not args.no_stdout:
        print(content)

    path = Path("pyproject.toml")

    data = tomlkit.parse(path.read_text())
    if "keywords" not in data["project"]:
        data["project"]["keywords"] = []
    
    
    print("keywords", data["project"]["keywords"]  )
    print(type(data["project"]["keywords"] ))
    kz = data["project"]["keywords"]
    kz = [k for k in kz if not k.startswith("hash:")]
    kz.append("hash:" + partial_hash)
    data["project"]["keywords"] = kz
    

    path.write_text(tomlkit.dumps(data))
    print(f"Hash mis à jour dans les metadata→ {partial_hash}")


"""
# Le hash du package est injecté  
from importlib.metadata import metadata
def get_package_hash(package_name):
    meta = metadata(package_name)
    keywords = meta.get_all('Keywords', [])
    # On cherche l'élément qui commence par "hash:"
    kz = keywords[0]
    for k in kz.split(','):
        if k.startswith("hash:"):
            return k.split(":", 1)[1]
    return None
 
get_package_hash("labquiz")
"""


if __name__ == "__main__":
    main()

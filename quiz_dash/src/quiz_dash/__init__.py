
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("quiz_editor")
except PackageNotFoundError:
    __version__ = "0.0.0"



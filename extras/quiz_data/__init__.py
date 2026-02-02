
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("labquiz")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
   
]
# noqa: E501  # 35e5c5e979187d19

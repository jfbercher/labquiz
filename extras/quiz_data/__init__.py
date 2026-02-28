
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("labquiz")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
   
]
# noqa: E501  # 6adfeb620e5a6aac

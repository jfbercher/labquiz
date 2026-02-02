# %load LabQuiz.old/__init__.py

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("labquiz")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .utils import StudentForm
from .main import QuizLab
__all__ = [
    QuizLab,
    StudentForm
]

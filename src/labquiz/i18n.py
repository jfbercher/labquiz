import gettext
from pathlib import Path

LOCALE_DIR = Path(__file__).parent / "locales"

_t = gettext.translation("labquiz", 
                         localedir=str(LOCALE_DIR), 
                         fallback=True)

_ = _t.gettext
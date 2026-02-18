import gettext
from pathlib import Path
import os
import locale

'''LOCALE_DIR = Path(__file__).parent / "locales"

_t = gettext.translation("labquiz", 
                         localedir=str(LOCALE_DIR), 
                         fallback=True)

_ = _t.gettext'''

import gettext
from pathlib import Path

# On définit le chemin des locales par rapport à ce fichier
LOCALE_DIR = Path(__file__).parent / "locales"

# Variable interne qui contient l'objet de traduction actuel
_current_trans = gettext.NullTranslations()

'''def set_language(lang_code):
    """Update the global translation object for the whole module."""
    global _current_trans
    _current_trans = gettext.translation(
        "messages", 
        localedir=LOCALE_DIR, 
        languages=[lang_code], 
        fallback=True
    )'''

def get_available_languages():
    """Return a list of available languages."""
    return [
        d.name for d in LOCALE_DIR.iterdir() 
        if d.is_dir() and any(d.rglob("*.mo"))
    ]

def get_best_language():
    # 1. Check environment variables (robust method)
    for env_var in ('LANGUAGE', 'LC_ALL', 'LC_MESSAGES', 'LANG'):
        lang = os.environ.get(env_var)
        if lang:
            # Nettoyage : 'fr_FR.UTF-8' -> 'fr'
            return lang.split('_')[0].split('.')[0].lower()

    # 2. If nothing is found, use the locale module
    try:
        lang, _ = locale.getdefaultlocale()
        if lang:
            return lang.split('_')[0].lower()
    except:
        pass

    return "en" # Fallback

def set_language(lang_code):
    global _current_trans
    available = get_available_languages()
    
    if lang_code not in available:
        print(f"Warning: Language '{lang_code}' not found. Available: {available}")
        print("Defaulting to 'en'.")

    _current_trans = gettext.translation(
        "labquiz", 
        localedir=LOCALE_DIR, 
        languages=[lang_code], 
        fallback=True
    )


def _(message):
    """Dynamic wrapper that always uses the current language."""
    return _current_trans.gettext(message)

def auto_init_lang():
    detected = get_best_language()
    available = get_available_languages()
    if detected in available:
        set_language(detected)
    else:
        # Si la langue détectée n'est pas traduite, on met l'anglais
        set_language("en")

# Automatic initialization when loading the module
auto_init_lang()

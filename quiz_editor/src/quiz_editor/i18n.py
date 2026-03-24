from pathlib import Path
import gettext
import streamlit as st


def get_localedir():
    """
    Constructs an absolute path to ./locales based on the actual location of the file.
    """
    return Path(__file__).resolve().parent / "locales"


def get_translator(lang: str):
    """
    Returns a _() function for the requested language.
    """
    localedir = get_localedir()

    translation = gettext.translation(
        domain="quiz_editor",
        localedir=localedir,
        languages=[lang],
        fallback=True,
    )

    return translation.gettext


def init_i18n(default_lang="en"):
    """
    Initialize translation in st.session_state and returns the function _.
    """
    if "lang" not in st.session_state:
        st.session_state.lang = default_lang

    if "translator" not in st.session_state:
        st.session_state.translator = get_translator(st.session_state.lang)

    return st.session_state.translator


def set_language(lang: str):
    """
    Change language dynamically.
    """
    st.session_state.lang = lang
    st.session_state.translator = get_translator(lang)
    return st.session_state.translator

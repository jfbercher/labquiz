import streamlit as st
from ruamel.yaml import YAML
import sys
import os
import re
import types
import copy 

# Simulateur de f-strings
import numpy as np # For simulation in the validator
import pandas as pd # For simulation in the validator

import random
import ast  #For simulation in the validator
from pathlib import Path
from io import StringIO
from collections import Counter
import ast
from convert_quiz_format import convert_quiz_data_v1_to_v2
#from i18n import _

from i18n import init_i18n, set_language, get_translator
_ = init_i18n(default_lang="en")
rng = np.random.default_rng()

# --- YAML CONFIGURATION  ---
# 
yaml = YAML()
yaml.preserve_quotes = True 
yaml.indent(mapping=2, sequence=4, offset=2)



def select_language():
    global _
    # Language selection
    #lang = st.sidebar.selectbox("Language", ["🇬🇧 en", "🇫🇷 fr"], index=["en", "fr"].index(st.session_state.lang))
    languages = {
        "en": "🇬🇧 English",
        "fr": "🇫🇷 Français",
        "es": "🇪🇸 Spanish",
        "it": "🇮🇹 Italian",
        "de": "🇩🇪 German",
        "ar": "🇪🇨 Arabic",
        "cn": "🇨🇳 Chinese",
    }

    lang = st.sidebar.selectbox(
        "Language",
        options=list(languages.keys()),
        format_func=lambda x: languages[x],
        index=list(languages.keys()).index(st.session_state.lang),
    )

    if lang != st.session_state.lang:
        #print(f"Language changed from {st.session_state.lang} to {lang}")
        if "quiz_title" in st.session_state and st.session_state.quiz_title == _("New Quiz"): 
            _ = set_language(lang)
            st.session_state.quiz_title = _("New Quiz")
        else:
            _ = set_language(lang)
        st.rerun()

    return lang, _

def apply_custom_styles():

    global _
    st.markdown("""
    <style>

    /* ===== File Uploader compact ===== */
    div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {{
        padding-top: 0.3rem;
        padding-bottom: 0.3rem;
        min-height: unset;
    }}
    div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] div[data-testid="stFileUploaderDropzoneInstructions"] {{
        display: none;
    }}
    div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] span[data-testid="stBaseButton-secondary"] {{
        margin: 0 auto;
    }}
    div[data-testid="stFileUploader"] ul {{
        display: none;
    }}

    /* ===== Main title ===== */
    input[aria-label="{doctitle}"] {{
        font-size: 1.75rem !important;
        font-weight: 600 !important;
        border: none !important;
        background: transparent !important;
    }}
    input[aria-label="{doctitle}"]:focus {{
        outline: none !important;
        box-shadow: none !important;
    }}

    /* ===== Sidebar ===== */
    section[data-testid="stSidebar"] {{
        padding-top: 0rem;
    }}
    section[data-testid="stSidebar"] .block-container {{
        gap: 0.35rem;
    }}
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
        margin-bottom: 0.25rem;
    }}
    section[data-testid="stSidebar"] hr {{
        margin-top: 0.25rem;
        margin-bottom: 0.25rem;
    }}
    [data-testid="stSidebarHeader"] {{
        height: 1.0rem;
        min-height: 0.5rem;
        padding: 0;
        margin: 0;
    }}
    [data-testid="stLogoSpacer"] {{
        display: none;
    }}
    [data-testid="stSidebarCollapseButton"] {{
        margin-top: 0;
        padding: 0;
    }}

    /* ===== Header principal ===== */
    header[data-testid="stHeader"] {{
        height: 2rem;
        min-height: 2rem;
        padding: 0;
        margin: 0;
    }}
    div[data-testid="stToolbar"] {{
        padding: 0;
        margin: 0;
        height: 2rem;
    }}
    div[data-testid="stToolbarActions"],
    div[data-testid="stAppDeployButton"],
    span[data-testid="stMainMenu"] {{
        margin-top: 0;
        padding-top: 0;
    }}

    /* ===== Container principal ===== */
    .block-container {{
        padding-top: 0rem;
        padding-bottom: 0.5rem;
    }}

    </style>
    """.format(doctitle=_("Document Title")), unsafe_allow_html=True)


def init_session_states(FILE_PATH):

    if "shared_fn" not in st.session_state:
        st.session_state["shared_fn"] = FILE_PATH

    if "fn_sidebar" not in st.session_state:
        st.session_state["fn_sidebar"] = st.session_state["shared_fn"]
    if "fn_main" not in st.session_state:
        st.session_state["fn_main"] = st.session_state["shared_fn"]

    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = None

    if "output_content" not in st.session_state:
        st.session_state.output_content = None

    # Session initialization for data persistence
    if 'data' not in st.session_state:
        st.session_state.data = load_data(FILE_PATH)

    if "quiz_title" not in st.session_state:
        st.session_state.quiz_title = st.session_state.data.get("title", _("Enter a title here"))

    # For exports
    if 'selected_for_export' not in st.session_state:
        # On initialise avec False pour tous les quiz existants
        st.session_state.selected_for_export = {}


def get_and_prepare_data():
    
    # sort data and resave it
    data = st.session_state.data
    data = {k: data[k] for k in sorted(data, key=natural_key)} #extract_key_number)}
    st.session_state.data = data

    # Then extracts categories and tags
    quiz_ids_all = [k for k in data.keys() if k != 'title']

    # We collect everything in one go
    all_categories = set()
    all_tags = set()
    cat_counts = Counter()

    for qid in quiz_ids_all:
        cat = data[qid].get('category', _('None'))
        all_categories.add(cat)
        cat_counts[cat] += 1
        
        tags = data[qid].get('tags', [])
        if isinstance(tags, list):
            all_tags.update(tags)

    if _("None") in all_categories:
        all_categories.remove(_('None')) #insert(0, all_categories.pop(all_categories.index("Aucune")))
    # Set up sorted lists for the widgets
    sorted_cats = sorted(list(all_categories))
    sorted_tags = sorted(list(all_tags))

    if "current_quiz" not in st.session_state:
        # We initialize with the first quiz in the list, if there is one.
        st.session_state.current_quiz = quiz_ids_all[0] if quiz_ids_all else None

    return data, quiz_ids_all, cat_counts, sorted_cats, sorted_tags

def trigger_rerun():
    # This function does nothing other than force Streamlit 
    # to reread the entire script with the new values.
    pass


def sync_export(qid, source_key):
    # We retrieve the value of the widget that has just changed

    val = st.session_state[source_key]
    # We update the reference dictionary
    st.session_state.selected_for_export[qid] = val
    
    # We synchronize the other widget so that it is up to date on the next display
    other_key = f"check_side_{qid}" if "main" in source_key else f"check_main_{qid}"
    st.session_state[other_key] = val

def validate_fstring(text):
    if not text or '{' not in text:
        return None
    
    start = 0
    while True:
        start = text.find('{', start)
        if start == -1: break
        
        end_brace = text.find('}', start)
        if end_brace == -1: return _("⚠️ Unclosed brace")
            
        full_block = text[start+1:end_brace].replace('\n', '')
        
        # We try to isolate the 'code' part from the 'formatting' 
        # We are looking for the rightmost ":" which is not in parentheses
        code_part = full_block
        bracket_level = 0
        for i in range(len(full_block)-1, -1, -1):
            char = full_block[i]
            if char == ')': bracket_level += 1
            elif char == '(': bracket_level -= 1
            elif char == ':' and bracket_level == 0:
                code_part = full_block[:i]
                break
        
        try:
            ast.parse(code_part.strip(), mode='eval')
        except SyntaxError as e:
            return f"⚠️ Error in '{{{code_part.strip()}}}' : {e.msg}"
            
        start = end_brace + 1
    return None

# Preview LaTeX
def render_preview(label, text, context=None):
    """Displays a preview if LaTeX or Markdown is present."""
    global _
    from convert_utils import evaluate_fstring, evaluate_text

    if text and ('$' in text or '{' in text):
        with st.container():
            #st.caption(f"Aperçu du rendu ({label}) :")
            st.markdown(
                f"<div style='font-size:0.8rem; color:gray; margin-bottom:-0.2rem;'>" \
                + _("Render preview ({label}) :</div>").format(label=label),
                unsafe_allow_html=True
            )
            if context:
                #if not '{' in text: text = f'{{ {text} }}'
                text = evaluate_text(text, context)
            st.info(text) # st.info renders Markdown and LaTeX between $ natively

def help_button(title, content, key):
    @st.dialog(title)
    def show():
        st.markdown(content)
    if st.button("❓", key=key):
        show()

def clean_constraints(data_dict, copy=False):
    """Cleaning Empty Constraints"""
    if copy: data_dict = copy.deepcopy(data_dict)
    for q in list(data_dict.keys()):
        obj = data_dict[q]
        if isinstance(obj, dict) and 'constraints' in obj and not obj['constraints']:
            del obj['constraints']
    return data_dict

def save_my_yaml(filename):
    global _
    global yaml
    
    if not filename:
        st.error(_("The file name cannot be empty."))
        return
    
    # Cleaning Empty Constraints
    data_to_save = st.session_state.data
    data_to_save = clean_constraints(data_to_save, copy=False)
    
    with open(filename, 'w', encoding='utf-8') as f:
        yaml.dump(data_to_save, f)
    
    st.success(_("File saved successfully as `{filename}`").format(filename=filename))
    return data_to_save

def get_quiz_yaml_string(data_dict):
    """Pure function to get YAML string."""
    global yaml
    from io import StringIO
    stream = StringIO()
    data_dict = clean_constraints(data_dict, copy=False)
    yaml.dump(data_dict, stream)
    return stream.getvalue()


def update_from_sidebar():
    val = st.session_state["fn_sidebar"]
    st.session_state["shared_fn"] = val
    st.session_state["fn_main"] = val

def update_from_main():
    val = st.session_state["fn_main"]
    st.session_state["shared_fn"] = val
    st.session_state["fn_sidebar"] = val


def load_data(FILE_PATH):
    global _
    if not os.path.exists(FILE_PATH):
        return {"title": _("New Quiz"), "quiz1": {"question": "", "propositions": []}}
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = yaml.load(f)
        data = convert_quiz_data_v1_to_v2(data)
        return data
    
def prepare_data(indata, output_file, mode="crypt", pwd=""):    
    from labquiz.putils import crypt_data, encode_data
    import copy

    global _

    # Lecture du fichier YAML
    data = copy.deepcopy(indata)

    # Suffle propositions for each quiz
    for quiz_name, quiz_content in data.items():
        if "propositions" in quiz_content:
            random.shuffle(quiz_content["propositions"])
            
    # Questions only
    data_only = copy.deepcopy(data)
    # removing hints for every quiz
    for quiz_name, quiz_content in data_only.items():
        if quiz_name == "title": continue 
        quiz_content.pop("constraints", None)
        for prop in quiz_content["propositions"]:
            keys_to_remove = {"expected", "answer", "tip"}
            for k in keys_to_remove:
                prop.pop(k, None)

    path = Path(output_file)
    if mode == "crypt":
        data_out = crypt_data(data, f"{path.stem}_crypt", pwd=pwd)
        data_only_out = crypt_data(data_only, f"{path.stem}_qo_crypt",  pwd=pwd)
    elif mode == "enc":
        data_out = encode_data(data)
        data_only_out = encode_data(data_only) 
    else:
        mode = "yml"
        data_out = data
        data_only_out = data_only
        
    if mode == "crypt" and pwd != '':
        st.warning(_("⚠️ File encrypted with password. Be sure to use the `mandatoryInternet=True` option when initializing the quiz")) 
        #("⚠️ File crypted with pwd. Ensure to use the `mandatoryInternet=True` option in quiz init")   
        #st.warning("⚠️ File crypted with pwd. Ensure to use the `mandatoryInternet=True` option in quiz init")    
    
    return data_out, data_only_out

def extract_key_number(key):
    match = re.search(r'(\d+)$', key)
    return int(match.group(1)) if match else 0

def natural_key(string_): #Gemini
    """Splits the string into a list of strings and integers."""
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', string_)]

def build_yaml(fname):
    data_to_save = save_my_yaml(fname)
    stream = StringIO()
    yaml.dump(data_to_save, stream)
    return stream.getvalue()

## Functions for templates variables

def is_valid_identifier(name):
    """Check whether a string is a valid Python variable name."""
    return re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name)

def safe_eval(expr):
    """
    Evaluate expression in a restricted namespace.
    Only rng, numpy and pandas are allowed.
    """
    return eval(expr, {"__builtins__": {}}, {"rng": rng, "np": np, "pd": pd})

# --- 1. Suggestion Helper ---
def get_default_suggestion(var_type, var_structure, engine_display):
    """Logic to suggest a default string based on type/structure/engine."""
    #print("default_suggestion", "var_type", var_type, "var_structure", var_structure, "engine_display", engine_display)
    size = 1 if var_structure == "scalar" else 3
    if engine_display == "numpy rng.":
        if var_type == "int":
            if size==1:
                return f"integers(0, 10)"
            return f"integers(0, 10, size={size})"
        elif var_type == "float":
            if size==1:
                return f"normal(0, 1)"
            return f"normal(0, 1, size={size})"
        return f"normal(0, 1, size={size})"
    else:  # pandas
        if "DataFrame" in var_structure:
            if var_type == "int":
                return "DataFrame(rng.integers(0, 10, (3,2)))"
            else:
                return "DataFrame(rng.normal(0, 1, (3,2)))"
        else:
            if var_type == "int":
                return "Series(rng.integers(0, 10, 3))"
            else:
                return "Series(rng.normal(0, 1, 3))"

# --- 2. Callback for Automatic Updates ---
def update_suggestion_callback(q_id, row_id):
    """Triggered when Type, Structure, or Engine changes to update the Call field."""
    type_key = f"type_{q_id}_{row_id}"
    struct_key = f"struct_{q_id}_{row_id}"
    engine_key = f"engine_{q_id}_{row_id}"
    call_key = f"call_{q_id}_{row_id}"
    
    # Update the call value in session_state immediately
    st.session_state[call_key] = get_default_suggestion(
        st.session_state[type_key],
        st.session_state[struct_key],
        st.session_state[engine_key]
    )

def on_change_update_and_save(q_data, q_id, row_id, update_suggestion=True):
    if update_suggestion: update_suggestion_callback(q_id, row_id)
    varname = st.session_state[f"name_{q_id}_{row_id}"]
    old_varname = st.session_state[f"name_{q_id}_{row_id}_old"]
    if varname != old_varname:
        # varname has changed - delete the old one
        st.session_state[f"name_{q_id}_{row_id}_old"] = varname
        if old_varname in q_data.get("variables", {}):
            del q_data["variables"][old_varname]
    if is_valid_identifier(varname):
        var_dict = {
        'type' : st.session_state.get(f"type_{q_id}_{row_id}", "int"),
        "structure": st.session_state.get(f"struct_{q_id}_{row_id}", "scalar"),
        "engine": st.session_state.get(f"engine_{q_id}_{row_id}", "numpy rng."),
        }
        var_dict["call"] = st.session_state.get(f"call_{q_id}_{row_id}",
            get_default_suggestion( var_dict["type"], var_dict["structure"], var_dict["engine"]))
        if q_data.get("variables") is None:
            q_data["variables"] = {}
        q_data["variables"][varname] = var_dict

def to_python(obj):
    # For serialization: converts pandas and numpy objects to python types
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    return obj

## Exports

@st.dialog(_("Configure & Export"))
def export_config_dialog(data, selected_qids, format_type):
    global _

    st.subheader(f"Format : {format_type}")
    
    # Common field: Filename
    default_name = _("my_quiz")
    file_name = st.text_input(_("File name (without extension)"), value=default_name)
    file_name = Path(file_name).stem
    reindex_labels = st.checkbox(_("Reindex question labels"), 
                    help=_("Reindex to ensure continuity of labels, e.g.: quiz1, quiz2, etc."), value=False)

    export_data = {"title": data.get('title', _('Untitled'))}    
    for i, old_id in enumerate(selected_qids):
        key = f"quiz{i+1}" if reindex_labels else old_id
        export_data[key] = copy.deepcopy(data[old_id])

    
    if format_type == _("Extract (YAML)"):
        fmt = st.radio(_("Export Type"), [_("Encrypted YAML"), _("Encoded YAML"), _("Unencoded YAML")])
        pwd = ""
        if fmt == _("Encrypted YAML"):
            col_crypt1, col_crypt2  = st.columns([6, 4])
            with col_crypt1:
                pwd = st.text_input(_("Password"), type="password", help=_("💡 This pwd participates in encryption"))
        
        mode = "crypt" if fmt == _("Encrypted YAML") else ("enc" if fmt == _("Encoded YAML") else "yml")
        outdata, outdata_only = prepare_data(export_data, file_name, mode=mode, pwd=pwd)

        st.info(_("💡 This mode allows you to save all or part of the questions in a new file, ") \
        + _("with optional encryption."))

        col1, col2 = st.columns(2)
        col1.download_button(_("📥 Full Quiz"), data=get_quiz_yaml_string(outdata), 
                           file_name=f"{file_name}.yaml", mime="text/yaml")
        col2.download_button(_("📥 Questions Only"), data=get_quiz_yaml_string(outdata_only), 
                           file_name=f"{file_name}_qo.yaml", mime="text/yaml")

    if format_type == _("Interactive (self-assessment)"):
        from convert_to_interactive_html import convert_to_interactive_html
        output_content = convert_to_interactive_html(export_data, lang=st.session_state.lang)
        st.info(_("💡 This mode allows immediate self-correction for students."))
        st.download_button(_("⬇️ Download HTML"), data=output_content, file_name=f"{file_name}.html", mime="text/html")

    elif format_type == _("Exam (Server)"):
        server_url = st.text_input(_("Receiving server URL"), placeholder="https://script.google.com/...")
        if server_url:
            from convert_to_html_exam import convert_to_server_quiz
            output_content = convert_to_server_quiz(export_data, server_url, lang=st.session_state.lang)
        else:
            st.warning(_("Please enter the server URL to continue."))
        st.download_button(_("⬇️ Download HTML"), data=output_content, file_name=f"{file_name}.html", mime="text/html")

    elif format_type == "AMC (LaTeX)":
        neg_points = st.checkbox(_("Negative points (-1 malus)"), value=True)
        from amc_exporter import convert_to_amc_latex
        output_content = convert_to_amc_latex(export_data, use_negative_points=neg_points)
        st.info(_("💡 Extraction in LaTeX-AMC format."))
        st.download_button(_("⬇️ Download LaTeX-AMC"), data=output_content, file_name=f"{file_name}.tex", mime="text/latex")


def render_template_editor(q_id, q_data, lang_func):
    """
    Handles the UI and logic for template variables (np.random, pandas, etc.)
    extracted from the main loop for clarity.
    """
    _ = lang_func
    import numpy as np
    import pandas as pd
    
    # Local RNG instance for previews
    rng = np.random.default_rng()

    # Initialization of rows in session_state for this specific question
    if q_id not in st.session_state:
        st.session_state[q_id] = types.SimpleNamespace()
    
    variables_dict = q_data.get("variables", {})
    # Sort and synchronize session_state with loaded data
    variables_dict = {k: variables_dict[k] for k in sorted(variables_dict)}
    st.session_state[q_id].rows = list(range(len(variables_dict)))
    
    for idx, (name, config) in enumerate(variables_dict.items()):
        # Pre-fill widgets keys to ensure persistence
        st.session_state[f"name_{q_id}_{idx}"] = name
        st.session_state[f"name_{q_id}_{idx}_old"] = name
        for key in ['type', 'struct', 'engine', 'call']:
            field_key = f"{key}_{q_id}_{idx}"
            if field_key not in st.session_state:
                st.session_state[field_key] = config.get(key if key != 'struct' else 'structure', "")

    st.subheader(_("Template variables"))
    
    # Header with Help and Add buttons
    cols_header = st.columns([0.06, 0.24, 0.7], gap="small")
    with cols_header[0]:
        help_button(_("Template variables"), 
                    _("Define variable names and the rule to generate their values. "
                      "These variables can be used in the question text or proposals using "
                      "the Python f-string syntax: {var_name}. "
                      "For HTML export, they are generated once per user. "
                      "For LaTeX-AMC export, they are used to generate the requested number of versions."), 
                    key=f"help_var_{q_id}")
    with cols_header[1]:
        if st.button(_("➕ Add variable"), key=f"btn_add_{q_id}"):
            new_idx = max(st.session_state[q_id].rows) + 1 if st.session_state[q_id].rows else 0
            st.session_state[q_id].rows.append(new_idx)
            st.session_state[f"call_{q_id}_{new_idx}"] = get_default_suggestion("int", "scalar", "numpy rng.")
            #st.rerun()
    with cols_header[2]:
        if st.button(_("🔁 Regenerate"), key=f"btn_resetVars_{q_id}"):
           st.rerun()
        

    active_vars = {}
    # Render table-like rows
    for row_id in st.session_state[q_id].rows:
        cols = st.columns([1.5, 1, 1.2, 1.2, 2.5, 0.6])
        
        key_name = f"name_{q_id}_{row_id}"
        # Key for storing the old value
        if f"{key_name}_old" not in st.session_state:
            st.session_state[f"{key_name}_old"] = ""
        var_name = cols[0].text_input("Name", key=f"name_{q_id}_{row_id}", label_visibility="collapsed",
                                     on_change=on_change_update_and_save, args=(q_data, q_id, row_id)) 
        
        # Selectboxes for Type, Structure, and Engine
        v_type = cols[1].selectbox("Type", ["int", "float"], key=f"type_{q_id}_{row_id}", label_visibility="collapsed",
                                   on_change=on_change_update_and_save, args=(q_data, q_id, row_id))
        v_struct = cols[2].selectbox("Structure", ["scalar", "list", "numpy array", "pandas Series", "pandas DataFrame"], 
                                     key=f"struct_{q_id}_{row_id}", label_visibility="collapsed",
                                     on_change=on_change_update_and_save, args=(q_data, q_id, row_id))
        v_eng = cols[3].selectbox("Engine", ["numpy rng.", "pandas."], key=f"engine_{q_id}_{row_id}", label_visibility="collapsed",
                                  on_change=on_change_update_and_save, args=(q_data, q_id, row_id))
        v_call = cols[4].text_input("Call", key=f"call_{q_id}_{row_id}", label_visibility="collapsed",
                                   on_change=on_change_update_and_save, args=(q_data, q_id, row_id, False))

        if cols[5].button("❌", key=f"del_{q_id}_row{row_id}"):
            st.session_state[q_id].rows.remove(row_id)
            q_data.get("variables", {}).pop(var_name, None)
            st.rerun()

        # Preview Logic
        if var_name and is_valid_identifier(var_name):
            active_vars[var_name] = {"type": v_type, "structure": v_struct, "engine": v_eng, "call": v_call}
            prefix = "rng." if v_eng == "numpy rng." else "pd."
            try:
                result = eval(f"{prefix}{v_call}", {"__builtins__": {}}, {"rng": rng, "np": np, "pd": pd})
                active_vars[var_name]["preview"] = to_python(result)
                st.caption(f"Preview: `{var_name} = {result}`")
            except Exception as e:
                st.error(f"Error: {e}")

    # Update global data object
    q_data['variables'] = active_vars

def render_propositions_editor(q_id, q_data, lang_func):
    """
    Renders the UI for editing question proposals.
    Dynamically switches 'expected' field between checkbox (standard) 
    and text_input (templates) with f-string validation.
    """
    _ = lang_func
    import copy

    st.markdown(f"### {_('Propositions')}")

    if 'propositions' not in q_data:
        q_data['propositions'] = []

    context = None
    # Check if we are in a template mode
    is_template = q_data.get('type') in ['mcq-template', 'numeric-template']
    if is_template:
        variables = q_data['variables']
        context = {var_name: variables[var_name]['preview'] for var_name in variables.keys()}

    for i, p in enumerate(q_data['propositions']):
        with st.expander(f"Proposition {i+1} : {p.get('label', '...')}", expanded=True):
            # Ligne 1 : Label et Correct/Expected
            cb1, cb2, cb3 = st.columns([2, 2, 1])
                
            with cb1:
                p['label'] = st.text_input(f"{_('Label')} (p{i+1})", p.get('label', ''), key=f"lab_{q_id}_{i}", help=_("Short identifier for the proposition."))
                render_preview('label', p['label'])

            p['proposition'] = st.text_area(_("Proposal Content"), p.get('proposition', ''), key=f"prop_{q_id}_{i}", height=70)
            render_preview('proposition', p['proposition'], context)
            p['hint'] = st.text_input(_("Hint"), p.get('hint', ''), key=f"hin_{q_id}_{i}", help=_("Optional hint displayed if the user fails"))
            render_preview('hint', p['hint'], context)
            p['answer'] = st.text_area(_("Answer/Feedback"), p.get('answer', ''), key=f"ans_{q_id}_{i}", height=70, help=_("Feedback shown after validation"))
            render_preview('answer', p['answer'], context)
        
            with cb3:
                # --- DYNAMIC EXPECTED FIELD ---
                if not is_template:
                    # Standard mode: Boolean Checkbox
                    p['expected'] = st.checkbox(_("Correct?"), p.get('expected', False), key=f"exp_{q_id}_{i}")
                else:
                    # Template mode: Text Input with f-string validation
                    exp_val = p.get('expected', "x+y" if q_data.get('type') == 'numeric-template' else "x + y > 0")
                    expected = st.text_input(
                        _("Expected value"), 
                        value=str(exp_val), 
                        key=f"e_{q_id}_{i}",
                        on_change=trigger_rerun  
                    )
                    
                    # Direct validation of New_val 
                    val_error = validate_fstring(expected)
                    if val_error:
                        st.error(val_error)
                    else:
                        p['expected'] = expected
                        if not '{' in expected: expected = f'{{ {expected} }}'
                        render_preview('Expected', expected, context)




            # Bonus/Malus
            b_val = str(p.get('bonus', ''))
            m_val = str(p.get('malus', ''))
            
            
            # Logic for YAML cleaning
            cb1, cb2, cb3 = st.columns([2, 2, 1])
            with cb1:
                res_b = st.text_input(_("Bonus"), b_val, key=f"bon_{q_id}_{i}", help=_("Specific points if this is selected"))
                if res_b.strip(): 
                    p['bonus'] = int(res_b) if res_b.lstrip('-').isdigit() else res_b
                elif 'bonus' in p: 
                    del p['bonus']
            with cb2:    
                res_m = st.text_input(_("Malus"), m_val, key=f"mal_{q_id}_{i}", help=_("Points deducted if this is selected"))
                if res_m.strip(): 
                    p['malus'] = int(res_m) if res_m.lstrip('-').isdigit() else res_m
                elif 'malus' in p: 
                    del p['malus'] 
            with cb3:
                st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
                
                if st.button(_("🗑️ Delete"), key=f"del_{q_id}_{i}", use_container_width=True):
                    q_data['propositions'].pop(i)
                    st.rerun()

                if st.button(_("👯 Duplicate"), key=f"dup_prop_{q_id}_{i}", use_container_width=True):
                    new_prop = copy.deepcopy(p)
                    new_prop['label'] = f"{p.get('label', '')} " + _("copy")
                    q_data['propositions'].insert(i + 1, new_prop)
                    st.rerun()
            
            if q_data['type'] in ['numeric', 'numeric-template']:
                    cb1.text_input(_("Tolerance %"), str(p.get('tolerance', '')), key=f"tp_{q_id}_{i}", help=_("Tolerance in % (default = 0)."))
                    cb2.text_input(_("Tolerance Abs"), str(p.get('tolerance_abs', '')), key=f"ta_{q_id}_{i}", help=_("Tolerance in absolute value (default = 0.01)."))
                


    if st.button(_("➕ Add a proposal"), key=f"add_p_btn_{q_id}"):
        # Default initialization for new items
        new_item = {"label": f"p{len(q_data['propositions'])+1}", "proposition": "", "hint": "", "answer": ""}
    
        if is_template:
            new_item["expected"] = "x+y" if q_data.get('type') == 'numeric-template' else "x + y > 0"
            new_item["answer"] = "Solution: {x+y}" if q_data.get('type') == 'numeric-template' else "Solution: {x + y > 0}"
        else:
            new_item["expected"] = "3.14" if q_data.get('type') == 'numeric' else False
        q_data['propositions'].append(new_item)
        st.rerun()


def render_constraints_editor(q_id, q_data, lang_func):
    """
    Renders the logical constraints section (XOR, IMPLY, etc.) 
    between proposals.
    """
    _ = lang_func
    
    # 2. CONSTRAINTS
    with st.expander(_("🔗 Logical constraints (XOR, IMPLY...)"), expanded=False):
        if 'constraints' not in q_data: 
            q_data['constraints'] = []
        
        # Retrieving labels for multiselect
        available_labels = [p.get('label', f'p{i+1}') for i, p in enumerate(q_data.get('propositions', []))]
        constraints_types = ["XOR", "SAME", "IMPLY", "IMPLYFALSE"]
        
        # We use a copy of the list to iterate to avoid index issues during deletion
        for idx, c in enumerate(q_data['constraints']):
            cc1, cc2, cc3, cc4 = st.columns([1, 2, 1, 0.5])
            
            # Type selection
            current_type = c.get('type', 'XOR')
            type_idx = constraints_types.index(current_type) if current_type in constraints_types else 0
            c['type'] = cc1.selectbox(_("Type"), constraints_types, index=type_idx, key=f"ct_{q_id}_{idx}")
            
            # Labels selection (max 2)
            current_defaults = [i for i in c.get('indexes', []) if i in available_labels]
            c['indexes'] = cc2.multiselect(_("Labels (max 2)"), available_labels, default=current_defaults, max_selections=2, key=f"ci_{q_id}_{idx}")
            
            # Malus value
            c['malus'] = cc3.number_input(_("Malus"), value=int(c.get('malus', 1)), key=f"cm_{q_id}_{idx}")
            
            # Delete button with alignment
            cc4.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
            if cc4.button("🗑️", key=f"cdel_{q_id}_{idx}", help=_("Remove this constraint")):
                q_data['constraints'].pop(idx)
                st.rerun()
        
        # Add button (Note: added q_id to key to prevent conflicts if multiple questions)
        if st.button(_("➕ Add constraint"), key=f"add_c_btn_{q_id}"):
            q_data['constraints'].append({"indexes": [], "type": "XOR", "malus": 1})
            st.rerun()

def render_question_text(q_id, q_data, lang_func):
    """
    Renders the text area for the question and the type/label selectors.
    (But in template case, do not preview variables since they are defined afterwards)
    """
    _ = lang_func
    
    col_q1, col_q2 = st.columns([4, 1])

    with col_q1:
        q_data['question'] = st.text_area(
            _("Question text"), 
            q_data.get('question', ''), 
            height=80,
            key=f"text_area_{q_id}",
            help=_("Supports Markdown and LaTeX (e.g., $x^2$).")
        )
        render_preview("Question", q_data['question'], context=None)

    available_types = ["mcq", "numeric", "mcq-template", "numeric-template"]
    current_type = q_data.get('type', 'mcq')
    
    with col_q2:
        with st.container(border=True):
            q_data['type'] = st.selectbox(
                "Type", 
                available_types, 
                index=available_types.index(current_type) if current_type in available_types else 0,
                key=f"type_select_{q_id}",
                help=_("Defines the validation behavior of the quiz.")
            )

            q_data['label'] = st.text_input(
                _("Label"), 
                value=q_data.get('label', f'q:{q_id}'), 
                key=f"label_field_{q_id}",
                help=_("Question label.")
            )

def preview_question_text(q_id, q_data, lang_func):
    """
    Preview the text area for the question and the type/label selectors.
    """
    _ = lang_func
    
    context = None
    # Check if we are in a template mode
    is_template = q_data.get('type') in ['mcq-template', 'numeric-template']
    if is_template:
        try:
            variables = q_data['variables']
            context = {var_name: variables[var_name]['preview'] for var_name in variables.keys()}
        except:
            pass

    col_q1, col_q2 = st.columns([4, 1])

    with col_q1:
        render_preview("Question", q_data['question'], context)

def preview_expected(q_id, q_data, lang_func):
    """
    Preview the text area for the expected value in the case of templates.
    """
    _ = lang_func
    
    context = None
    # Check if we are in a template mode
    is_template = q_data.get('type') in ['mcq-template', 'numeric-template']
    if is_template:
        try:
            variables = q_data['variables']
            context = {var_name: variables[var_name]['preview'] for var_name in variables.keys()}
        except:
            pass

        col_q1, col_q2 = st.columns([4, 1])

        with col_q2:
            render_preview("Expected", q_data['expected'], context)



def render_taxonomy_editor(q_id, q_data, data, quiz_ids_all, sorted_tags, lang_func):
    """
    Handles Categories (dropdown + new entry) and Tags (multiselect).
    """
    _ = lang_func
    
    # Category field logic
    current_cat = q_data.get('category', _('None'))
    all_cats = sorted(list(set(data[qid].get('category', _('None')) 
                               for qid in quiz_ids_all 
                               if data[qid].get('category', _('None')) != _("None"))))
    menu_options = [_("None")] + all_cats + [_("➕ New category...")]
    
    default_idx = menu_options.index(current_cat) if current_cat in menu_options else 0

    col_cat1, col_cat2 = st.columns([4, 6])
    
    with col_cat1:
        selected_from_menu = st.selectbox(
            _("🗂️ Category"),
            options=menu_options,
            index=default_idx,
            key=f"select_cat_{q_id}",
            help=_("Question category"),
        )
        if selected_from_menu == _("➕ New category..."):
            new_cat_name = st.text_input(
                _("Name of the new category"),
                placeholder=_("Enter the name here..."),
                key=f"new_cat_input_{q_id}",
            )
            if new_cat_name and new_cat_name != current_cat:
                q_data['category'] = new_cat_name
                st.rerun()
        else:
            if selected_from_menu != current_cat:
                q_data['category'] = selected_from_menu
                st.rerun()

    with col_cat2:
        current_tags = q_data.get('tags', [])
        state_key = f"tags_state_{q_id}"

        if state_key not in st.session_state:
            st.session_state[state_key] = current_tags.copy()

        all_available_tags = sorted(set(sorted_tags) | set(st.session_state[state_key]))

        updated_tags = st.multiselect(
            _("🏷️ Tags (Difficulty, Sub-theme...)"),
            options=all_available_tags,
            default=st.session_state[state_key],   
            placeholder=_("Choose or enter a new tag"),
            accept_new_options=True,
            key=f"tags_widget_{q_id}"
        )

        if updated_tags != st.session_state[state_key]:
            st.session_state[state_key] = updated_tags
            q_data["tags"] = updated_tags


def render_question_header(q_id, filtered_ids, lang_func):
    """
    Renders the navigation bar and current question title.
    Based on lines 610-638 of your source.
    """
    _ = lang_func
    current_idx = filtered_ids.index(q_id)
    
    # Navigation and selection columns
    col_prev, col_title, col_next, col_check = st.columns([0.5, 3, 0.5, 1.5], vertical_alignment="center")

    with col_prev:
        if st.button("⬅️", disabled=(current_idx == 0), key=f"prev_{q_id}"):
            st.session_state.current_quiz = filtered_ids[current_idx - 1]
            st.rerun()

    with col_next:
        if st.button("➡️", disabled=(current_idx == len(filtered_ids) - 1), key=f"next_{q_id}"):
            st.session_state.current_quiz = filtered_ids[current_idx + 1]
            st.rerun()

    with col_title:
        st.subheader(_("Editing {current_quiz}").format(current_quiz=q_id))

    with col_check:
        main_key = f"check_main_{q_id}"
        # Sync with export selection state
        st.session_state[main_key] = st.session_state.selected_for_export.get(q_id, False)
        st.checkbox(
            _("Export"), 
            key=main_key,
            on_change=sync_export,
            args=(q_id, main_key),
            help=_("Check to include this question in future YAML exports")
        )



#-------------------------------------------------
#                      MAIN                      #
#-------------------------------------------------


def main():
    import copy
    global _

    # Menu to select (and change) language
    lang, _ = select_language()
    

    # Define the path of the YAML file
    if len(sys.argv) > 1 and sys.argv[1].endswith(('.yaml', '.yml')):
        FILE_PATH = sys.argv[1]
    else:
        FILE_PATH = "quiz.yaml"

    # Initialization of session variables
    init_session_states(FILE_PATH)
    # From data, extract all quizzes, categories and tags
    data, quiz_ids_all, cat_counts,sorted_cats, sorted_tags = get_and_prepare_data()
    

    # Set page config and apply custom styles
    st.set_page_config(layout="wide", page_title=_("YAML Editor - {FILE_PATH}").format(FILE_PATH=FILE_PATH),  
                    page_icon="src/quiz_editor/1F4C3.png") #📃")
    apply_custom_styles()
    

    # --- SIDEBAR SECTION ---

    data = st.session_state.data

    st.sidebar.title(_("📂 Browse"))
    st.sidebar.divider()
    st.sidebar.subheader(_("📥 Import new file"))


    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        _("Choose file"), 
        type=["yaml", "yml"],
        help=_("Loads the contents of a YAML file into the editor")
    )

    if uploaded_file is not None:
        #if st.sidebar.button("🚀 Charger ce fichier", use_container_width=True):
        if uploaded_file.name != st.session_state.last_uploaded_file:
            try:
                data = yaml.load(uploaded_file)
                data = convert_quiz_data_v1_to_v2(data) #precaution
                # 2. updatesession_state
                st.session_state.data = data
                st.session_state.data['title'] = data.get('title', _('📖 Enter a title here'))
                st.session_state["quiz_title"] = st.session_state.data["title"]
                # 3. Updating filename for future save
                st.session_state["shared_fn"] = uploaded_file.name
                st.session_state.last_uploaded_file = uploaded_file.name
                # 4. Success message and refresh
                st.toast(_("File `{file_name}` loaded successfully!").format(file_name=st.session_state['shared_fn']))
                st.rerun()
            except Exception as e:
                st.sidebar.error(_("Read Error:") + f"{e}")

    # QUIZ MANAGEMENT
    st.sidebar.divider()
    st.sidebar.subheader(_("🎰🎲♠ Quiz management"))

    # 3. Filtering widget in sidebar
    #quiz_ids = [k for k in data.keys() if k != 'title']
    quiz_ids_all = [k for k in data.keys() if k != 'title']

    # Category filter
    cat_options = [_("All ({len_all})").format(len_all=len(quiz_ids_all))] + [f"{c} ({cat_counts[c]})" for c in sorted_cats]
    selected_cat_ui = st.sidebar.selectbox(_("Category"), cat_options)
    all_cat_name =  _("All ({len_all})").split(" (")[0] #
    selected_cat = selected_cat_ui.split(" (")[0] if selected_cat_ui != all_cat_name else all_cat_name

    # Filtre Tags (utilise sorted_tags préparé plus haut)
    selected_tags = st.sidebar.multiselect(_("Filter by Tags"), 
                                placeholder=_("Choose a tag"), options=sorted_tags)

    ## Join filtering
    filtered_ids = []
    for qid in quiz_ids_all:
        q_cat = data[qid].get('category', _('None'))
        q_tags = data[qid].get('tags', [])
        
        # Category and tags check
        match_cat = (selected_cat == all_cat_name or q_cat == selected_cat)
        match_tags = all(tag in q_tags for tag in selected_tags)
    
        if match_cat and match_tags:
            filtered_ids.append(qid)

    ## Selection widget on filtered quizzes
    quiz_ids = filtered_ids
    selected_quiz = st.sidebar.selectbox(
        _("Choose a question from {lenf}").format(lenf=len(filtered_ids)), 
        quiz_ids, 
        index=quiz_ids.index(st.session_state.current_quiz) if st.session_state.current_quiz in quiz_ids else 0
        )
    if selected_quiz != st.session_state.current_quiz:
        st.session_state.current_quiz = selected_quiz
        st.rerun()

    ## Actions on selected quiz

    col1, col2 = st.sidebar.columns(2)
    ### 1. BOUTON CLONER
    with col1:
        if st.button(_("👯 Duplicate"), use_container_width=True, help="Copier ce quiz"):
            import copy
            numbers = [int(re.findall(r'\d+', k)[0]) for k in quiz_ids if re.findall(r'\d+', k)]
            next_num = max(numbers) + 1 if numbers else 1
            new_id = f"quiz{next_num}"
            st.session_state.data[new_id] = copy.deepcopy(data[selected_quiz])
            st.session_state.current_quiz = new_id
            st.rerun()
    ### 2. DELETE BUTTON WITH CONFIRMATION
    with col2:
        confirm_key = f"confirm_del_{selected_quiz}"
        if confirm_key not in st.session_state:
            st.session_state[confirm_key] = False

        if not st.session_state[confirm_key]:
            if st.button(_("🗑️ Delete"), use_container_width=True, help="Supprimer ce quiz"):
                st.session_state[confirm_key] = True
                st.rerun()
        else:
            # Confirmation request
            if st.button(_("❗ Confirm?"), use_container_width=True, type="primary"):
                if len(quiz_ids) > 1:
                    del st.session_state.data[selected_quiz]
                    remaining_ids = [k for k in st.session_state.data.keys() if k != 'title']
                    st.session_state.current_quiz = remaining_ids[0]
                    st.session_state[confirm_key] = False
                    st.rerun()
                else:
                    st.sidebar.error(_("Last quiz!"))
                    st.session_state[confirm_key] = False
            # Cancel button
            if st.button(_("Cancel"), use_container_width=True):
                st.session_state[confirm_key] = False
                st.rerun()
    
    ### 3. BUTTON NEW QUIZ - Create a new quiz
    available_types = ["mcq", "numeric", "mcq-template", "numeric-template"]

    with st.sidebar.popover(_("➕ New Quiz"), use_container_width=True):

        selected_type = st.selectbox(
            _("Select quiz type"),
            available_types,
            help=_("Defines the validation behavior of the quiz.")
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button(_("Create"), use_container_width=True):

                # Compute next quiz ID
                numbers = [
                    int(re.findall(r'\d+', k)[0])
                    for k in quiz_ids
                    if re.findall(r'\d+', k)
                ]
                next_num = max(numbers) + 1 if numbers else 1
                new_id = f"quiz{next_num}"

                # Create quiz structure depending on selected type
                if selected_type in ["numeric", "numeric-template"]:
                    expval = 3.14 if selected_type == "numeric" else "f'{a+b:.4f}'"
                    new_data = {
                        "type": selected_type,
                        "question": _("New question"),
                        "propositions": [
                            {"proposition": _("Choice 1"), "expected": expval}
                        ]
                    }

                elif selected_type in ["mcq", "mcq-template"]:
                    expval = False if selected_type == "mcq" else "f'{a} in {b}'"
                    new_data = {
                        "type": selected_type,
                        "question": _("New question"),
                        "propositions": [
                            {"proposition": _("Choice 1"), "expected": expval}
                        ]
                    }

                # Store quiz and update current selection
                st.session_state.data[new_id] = new_data
                st.session_state.current_quiz = new_id

                st.rerun()
        with col2:
            st.write("")  # Optional spacer (keeps layout symmetrical)


    # 4. Export section ---
    st.sidebar.divider()
    st.sidebar.title(_("📤 Export"))

    # Initialization of the selection dictionary
    if 'selected_for_export' not in st.session_state:
        st.session_state.selected_for_export = {qid: False for qid in quiz_ids}

    ## 4.1 CHECK/UNCHECK ALL BUTTONS
    col_all1, col_all2 = st.sidebar.columns(2)
    if col_all1.button(_("✅ Select all"), use_container_width=True):
        for qid in quiz_ids:
            st.session_state.selected_for_export[qid] = True
            # Force updating of widget keys
            st.session_state[f"check_side_{qid}"] = True
            st.session_state[f"check_main_{qid}"] = True
        st.rerun()

    if col_all2.button(_("❌ Deselect all"), use_container_width=True):
        for qid in quiz_ids:
            st.session_state.selected_for_export[qid] = False
            st.session_state[f"check_side_{qid}"] = False
            st.session_state[f"check_main_{qid}"] = False
        st.rerun()

    ## 4.2. SELECTION AREA (Checkboxes for each available quiz [may be already filtered])
    with st.sidebar.expander(_("Select questions"), expanded=False):
        for qid in quiz_ids:
            side_key = f"check_side_{qid}"
            # We ensure that the widget key exists 
            if side_key not in st.session_state:
                st.session_state[side_key] = st.session_state.selected_for_export.get(qid, False)

            st.checkbox(
                qid, 
                key=side_key,
                on_change=sync_export,
                args=(qid, side_key)
            )
            
    ## 4.3. Format choice menu
    selected_qids = [qid for qid in quiz_ids if st.session_state.selected_for_export.get(qid, False)]

    if selected_qids:
        st.sidebar.markdown(_("**{lsq} question(s) selected**").format(lsq=len(selected_qids)))
        
        # Preparing the export_data
        export_data = {"title": data.get('title', 'Sans titre')}
        for i, old_id in enumerate(selected_qids):
            export_data[f"quiz{i+1}"] = copy.deepcopy(data[old_id])

        # Export drop-down menu
        export_format = st.sidebar.selectbox(
            _("Choose export format"),
            ["---", _("Extract (YAML)"), _("Interactive (self-assessment)"), _("Exam (Server)"), "AMC (LaTeX)"]
        )

        if export_format != "---":
            if st.sidebar.button(_("Configure & Export"), use_container_width=True):
                export_config_dialog(data, selected_qids, export_format)
    else:
        st.sidebar.info(_("Check questions to export."))


    # END OF SIDEBAR SECTION

    #  ------------------------- MAIN AREA SECTION -------------------------

    with st.container():
        st.markdown(
            """
            <style>
            .block-container > div:first-child {
                margin-top: 1.0rem !important;
                padding-top: 1.0rem !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    title = st.text_input(
        _("Document Title"), 
        #value=f"📖 {st.session_state.data.get('title', 'Quiz sans titre')}",
        key="quiz_title",
        help=_("The main title of the YAML file (key ‘title’) - Editable here."),
    )
    data["title"] = title.lstrip("📖 ").strip()
    st.session_state.data["title"] = title.lstrip("📖 ").strip()
    #st.session_state["quiz_title"] = st.session_state.data["title"]
    st.caption(_("Current file: `{shared_fn}`").format(shared_fn=st.session_state['shared_fn']))
    
    col_save1, col_save2 = st.columns([6, 4])
    with col_save1:
        st.text_input(
        _("Edit to ‘Save As’."), 
        key="fn_main", 
        on_change=update_from_main,
        help=_("‘Save As’.")
        )

    with col_save2:
        st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
        btn_save, btn_download = st.columns([1, 1], gap="small")

        with btn_save:
            if st.button(_("💾 Save"), key="btn_main_save_final",width="stretch"):
                fname = st.session_state.get("fn_main") or st.session_state["shared_fn"]
                st.session_state.output_content = build_yaml(fname)
                st.toast(_("YAML cleaned and saved ✔️"))

        with btn_download:
            if st.session_state.output_content is not None:
                fname = st.session_state.get("fn_main") or st.session_state["shared_fn"]
                st.download_button(
                    label=_("Download!"),
                    data=st.session_state.output_content,
                    file_name=fname,
                    mime="text/yaml",
                    width="stretch",
                    help=_("Only AFTER saving the YAML")
                )

    if st.session_state.current_quiz:
        q_id = st.session_state.current_quiz
        q_data = data[q_id]
        
        st.divider()

    if st.session_state.current_quiz and filtered_ids:

        render_question_header(q_id, filtered_ids, _)
        render_taxonomy_editor(q_id, q_data, data, quiz_ids_all, sorted_tags, _)
        render_question_text(q_id, q_data, _)
                                
        # IF TEMPLATE: Call our new modular function
        if q_data['type'] in ['mcq-template', 'numeric-template']:
            render_template_editor(q_id, q_data, _)
            preview_question_text(q_id, q_data, _)

        # CONSTRAINTS
        render_constraints_editor(q_id, q_data, _)
        # PROPOSITIONS
        render_propositions_editor(q_id, q_data, _)
        
    else:
        st.info(_("Use the sidebar to navigate or create a new quiz."))

if __name__ == "__main__":
    main()
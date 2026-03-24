import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import time
from streamlit_autorefresh import st_autorefresh
from streamlit_local_storage import LocalStorage
#from localStorage import LocalStorage
import base64
import zlib
import zipfile
import io
import json 
import re
import markdown

#import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


# Labquiz functions import
from labquiz.main import QuizLab
from labquiz.putils import (
    readData, 
    check_integrity_msg, 
    check_hash_integrity, 
    correctQuizzesDf)
from labquiz.utils import calculate_quiz_score


# Global variable initialization
local_storage = None
# Others
verbose = False

from i18n import init_i18n, set_language, get_translator
_ = init_i18n(default_lang="en")


# --- 1. INITIAL RESTORATION (TEXTS ONLY) ---
def set_defaults():
    global _
    st.session_state["selected_lang"] = "en"
    st.session_state["lang"] = "en"
    st.session_state["url"] = ""
    st.session_state["secret"] = ""
    st.session_state["group"] = _('All')
    st.session_state["bareme_str"] = "{}"
    st.session_state["maxtries"] = 3
    st.session_state["seuil"] = 0.0 
    st.session_state["exam_title"] = "" 
    st.session_state["params_str"] = "{'retries':2, 'exam_mode':False, 'test_mode':False}"

    # --- OTHER INITIALIZATION ---
    if "df_results" not in st.session_state:
        st.session_state.df_results = None
    if "df_final" not in st.session_state:
        st.session_state.df_final = None
    st.session_state.show_scores = False
    if "last_correction_update" not in st.session_state:
        st.session_state.last_correction_update = None
    if "refresh_key" not in st.session_state:
        st.session_state.refresh_key = 0
    if "uploader_version" not in st.session_state:
        st.session_state.uploader_version = 0
    if "last_processed_file" not in st.session_state:
        st.session_state.last_processed_file = None
    if "FinalMarkScale" not in st.session_state:
        print("⚠️ Setting FinalMarkScale to 20 in set_defaults")
        st.session_state.FinalMarkScale = "20"
        st.session_state.TrueFinalMarkScale = "20"
    if "main_nav_state" not in st.session_state:
        st.session_state.main_nav_state = _("📡 Integrity Live")
    if "monitoring_nav_state" not in st.session_state:
        st.session_state.monitoring_nav_state = _("📊 Monitoring charts")
    if "correction_nav_state" not in st.session_state:
        st.session_state.correction_nav_state = _("🎯 Correction & Grades")

def sync(key):
    global local_storage 
    global _

    if local_storage is None:
        return # Security if local_storage is not ready yet
    try: 
        val = st.session_state[key] #st.session_state.get(key, None) #
    except:
        print("Syncing error for", key, "session state not present")
        return
    
    if key == "FinalMarkScale":  
        st.session_state["TrueFinalMarkScale"] = st.session_state[key]
    if key.startswith("quiz_file_"):
        st.session_state["quiz_file"] = val
        if verbose: print(key, val)
        if val is not None:
            content = val.getvalue()
            compressed = zlib.compress(content)
            encoded = base64.b64encode(compressed).decode()
            file_data = {
                "name": val.name,
                "b64": encoded
            }
            local_storage.setItem("file_package", json.dumps(file_data))
            # Empty the restored object to force use of the new widget
            st.session_state.pop("restored_file", None)
        else:
            # The user clicked on the widget cross
            local_storage.deleteItem("file_package")
            st.session_state.pop("restored_file", None)
    else:
        # Any other widget
        # Use json to store the value
        # print("stored", key, val)
        local_storage.setItem(key, json.dumps(val))

def perform_global_reset():
        global local_storage
        local_storage.deleteAll()
        while len(local_storage.storedItems) > 0:
            time.sleep(0.1)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.clear()
        st.session_state.uploader_version = int(time.time())
        set_defaults()
        print("Global reset done")
        st.rerun()


@st.cache_data(show_spinner=False)
def adhocReadData(url, secret, autorefresh, button_refresh):
    import time
    if verbose:print("Reading data...")
    time.sleep(0)
    tic = time.perf_counter()
    df, df_filt = readData(url, secret)
    toc = time.perf_counter()
    if verbose: print(f"Reading data execution time: {toc-tic:.3f} seconde(s)")
    return df, df_filt

def generate_cols_from_student(df, dropStudent=False):
    split_cols = df['student'].str.split(r'\s*,\s*', expand=True)
    split_cols.columns = ['name', 'firstname', 'class_group'][:split_cols.shape[1]]
    newdf = pd.concat([split_cols, df], axis=1)
    if dropStudent: newdf = newdf.drop(columns='student')
    return newdf

def recompute_score(adj_bareme, questions, exam_title, df_results, full_df, full_df_filt, quiz, seuil, final_weights, maxtries):

    coeffs = adj_bareme.loc["Coefficient"]
    res_copy = df_results.copy()
    
    if exam_title == "":
        # Scalar product
        res_copy["FinalMark"] = res_copy[questions].dot(coeffs) * (20 / sum(coeffs))
    else:
        # Complete calcul
        res_copy = correctQuizzesDf(
            data=full_df, data_filt=full_df_filt, quiz=quiz, 
            title=exam_title, seuil=seuil, weights=final_weights, 
            bareme=coeffs, maxtries=maxtries
        )
        res_copy["FinalMark"] = res_copy["Note"]
        res_copy.drop(columns='Note', inplace=True, errors='ignore')
        res_copy = generate_cols_from_student(res_copy, dropStudent=False)
    #print(res_copy.columns, res_copy.head(5))
    return res_copy

def apply_custom_styles():

    custom_css ="""
    <style>
    /* To raise the widget vertically a little (fragile)
    # and reduce vertical space around divider */
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] {
        margin-top: -0.85rem;
    }
    div[data-testid="stMarkdownContainer"] hr {
        margin-top: 0.3rem;
        margin-bottom: 0.3rem;
    }
    
    /*Remove sidebar header*/

    [data-testid="stLogoSpacer"] {
        display: none;
    }

    [data-testid="stSidebarHeader"] {
        height: auto;
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    
    /* Remove main area header */
    .block-container {
        padding-top: 0rem;
        margin-top: 1rem;
    }
    
    /* don't display streamlit_local_storage iframes */

    /* 1. Target the specific iframe by its title */
    iframe[title="streamlit_local_storage.st_local_storage"] {
        display: none !important;
        height: 0px !important;
        width: 0px !important;
        visibility: hidden !important;
    }

    /* 2. Go back to the parent to delete the reserved space (the slot) */
    div[data-testid="stElementContainer"]:has(iframe[title="streamlit_local_storage.st_local_storage"]) {
        display: none !important;
        margin-bottom: 0px !important;
        padding: 0px !important;
    }
    
    /* 3. Additional Security for Recent Streamlit Versions */
    div.element-container:has(iframe[title="streamlit_local_storage.st_local_storage"]) {
        display: none !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


#@st.cache_data
def prepare_monitoring_data(df):
    """
    Filters the dataframe to keep only the last valid attempts 
    before the correction was shown.
    """
    df_copy = df.copy()
    
    # Identify if a student has already seen the correction for a specific quiz
    df_copy.loc[:, "has_seen_correction"] = (
        df_copy["event_type"].eq("correction")
        .groupby([df_copy["student"], df_copy["quiz_title"]])
        .transform("cummax")
    ).fillna(False).astype(bool)

    # Filter to get the last 'validate' or 'validate_exam' event before correction
    df_last = (
        df_copy.query("(event_type == 'validate' or event_type == 'validate_exam') and not has_seen_correction", engine="python")
        .groupby(["student", "quiz_title"], as_index=False)
        .tail(1)
    )
    
    return df_last


def create_monitoring_plot(data, title, plot_type="student_counts", lang_func=None):
    """
    Generates a Plotly figure. 
    Supported types: 'student_counts', 'student_scores', 'class_results', 'hardest_quizzes', 'quizzes_selectivity'
    """
    _ = lang_func if lang_func else lambda x: x
    fig = go.Figure()
    
    # Default configuration
    orientation = 'h'
    
    # Data processing based on plot_type
    if plot_type == "student_counts":
        series = data.groupby("student")["quiz_title"].count().sort_index(ascending=False)
        color = "#3498db"
        x_label = _("Number of Quizzes")

    elif plot_type == "student_scores":
        series = data.groupby("student")["score"].sum().sort_index(ascending=False)
        color = "#f1c40f"
        x_label = _("Total Score")

    elif plot_type == "class_results":
        series = data["quiz_title"].value_counts().sort_index(
            key=lambda idx: idx.str.extract(r"(\d+)").fillna(0).astype(int)[0]
        )
        color = "#2ecc71"
        x_label = _("Number of Students")

    elif plot_type == "hardest_quizzes":
        series = data.groupby("quiz_title")["score"].mean().sort_values(ascending=False).tail(5)
        color = "#e74c3c"
        x_label = _("Average Score")

    elif plot_type == "quizzes_selectivity":
        nb_students = data['student'].nunique()
        series = (data.groupby("quiz_title")["score"].sum() / nb_students).sort_values(ascending=False)
        color = "#e74c3c"
        x_label = _("Selectivity Score")

    # Prepare labels: truncated for the axis, full for the hover tooltip
    full_labels = series.index.tolist()
    display_labels = [s[:15] + '..' if len(s) > 15 else s for s in full_labels]

    # Add trace with custom hover data
    fig.add_trace(go.Bar(
        y=display_labels, 
        x=series.values, 
        marker_color=color, 
        orientation=orientation,
        customdata=full_labels, # Store original names for hover
        hovertemplate="<b>%{customdata}</b><br>" + x_label + ": %{x}<extra></extra>"
    ))

    # Global formatting
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        margin=dict(l=10, r=20, t=50, b=50), 
        # Dynamic height based on number of items to prevent squashing
        # height=max(400, len(series) * 25), 
        template="plotly_white",
        
        # Y-axis configuration (Students / Quizzes names)
        yaxis=dict(
            tickfont=dict(size=10),
            automargin=True # Ensures truncated labels fit within the left margin
        ),
        
        # X-axis configuration (Numeric values)
        xaxis=dict(
            tickangle=0,
            # Force integer steps for count-based plots
            dtick=None,  #1 if plot_type in ["student_counts", "class_results"] else None,
            nticks=10,
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.3)'
        )
    )

    return fig


@st.dialog(_("Histogram of marks"))
def show_histogram(marks):
    fig = px.histogram(
        x=marks, 
        nbins=20,
        labels={'x': _("Marks"), 'y': _("Frequency")},
        title=_("Distribution of marks")
    )
    
    # Updated layout for aesthetics
    fig.update_layout(
        bargap=0.1, # Espace entre les barres
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title=_("Marks"),
        yaxis_title=_("Frequency")
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_student_session_track(student_data, student_name):
    
    # Real elapsed time from the start (in minutes)
    start_time = student_data['timestamp'].min()
    student_data['real_min'] = (student_data['timestamp'] - start_time).dt.total_seconds() / 60
    
    # 1. Build the compressed X axis
    threshold = 60 # 1 hour break
    shunted_gap = 15 
    
    plot_x = []
    current_plot_x = 0.0
    pause_positions = []
    
    # We store the real values to use as labels on the X axis
    tick_vals = [0.0]
    tick_texts = ["0"]
    # Logic for both Axis labels and Stem labels
    last_tick_pos = -100
    last_stem_label_pos = -100
    
    # Thresholds (adjust based on your visual preference)
    min_dist_ticks = 2 
    min_dist_labels = 2 # q1, q2 often take more horizontal space than numbers
    
    stem_labels = [] # To store q1, q2... or ""
    tick_vals = []
    tick_texts = []

    # Calcul des limites de l'axe Y avec une petite marge (10%)
    y_min = student_data["score"].min()
    y_max = student_data["score"].max()

    # On s'assure que l'axe inclut toujours au moins [0, 1] mais s'étend si besoin
    plot_y_min = min(-0.1, y_min - 0.1)
    plot_y_max = max(1.1, y_max + 0.2)


    for i in range(len(student_data)):
        # (Relative positioning in the x axis based on elapsed time)
        real_delta = student_data.iloc[i]['real_min'] - (student_data.iloc[i-1]['real_min'] if i > 0 else 0)
        if i > 0 and real_delta > threshold:
            pause_positions.append(current_plot_x + (shunted_gap / 2))
            current_plot_x += shunted_gap
        elif i > 0:
            current_plot_x += real_delta
        plot_x.append(current_plot_x)

        # 1. Logic for X-Axis Labels (the numbers)
        if current_plot_x - last_tick_pos > min_dist_ticks:
            tick_vals.append(current_plot_x)
            tick_texts.append(str(int(student_data.iloc[i]['real_min'])))
            last_tick_pos = current_plot_x
        
        # 2. Logic for Stem Labels (q1, q2...)
        short_name = "q" + re.search(r"(\d+)", str(student_data.iloc[i]["quiz_title"])).group(1)
        if current_plot_x - last_stem_label_pos > min_dist_labels:
            stem_labels.append(short_name)
            last_stem_label_pos = current_plot_x
        else:
            stem_labels.append("") # Hide the label if too close

    hover_data = list(zip(
        student_data["timestamp"].dt.strftime('%H:%M'),
        student_data["real_min"].round(1),
        student_data["quiz_title"],
        student_data["timestamp"].dt.strftime('%d/%m/%Y')
    ))

    fig = go.Figure()

    # 3. Stem Plot
    fig.add_trace(go.Scatter(
        x=plot_x,
        y=student_data["score"],
        mode='markers+text',
        name="Student Score",
        # On utilise la liste nettoyée 'stem_labels' au lieu de la liste complète
        text=stem_labels, 
        # On garde l'alternance top/bottom right pour dégager la vue
        textposition= "top center", #["top right", "bottom right"] * (len(student_data) // 2 + 1),
        textfont=dict(size=9),
        marker=dict(
            size=12, 
            color="#3498db", 
            line=dict(width=1, color='white')
        ),
        error_y=dict(
            type='data', 
            symmetric=False, 
            array=[0] * len(student_data), 
            arrayminus=student_data["score"], 
            color="#3498db", 
            thickness=1.5,
            width=0
        ),
        #customdata=student_data["timestamp"].dt.strftime('%H:%M'),
        ## Note : on affiche la vraie minute (real_min) dans le hover même si le label est masqué
        #hovertemplate="<b>Quiz: %{customdata}</b><br>Elapsed: %{text} min<br>Score: %{y:.2f}<extra></extra>"
        customdata=hover_data,
        # Construction du template :
        # customdata[3] = Day/Month/Year
        # customdata[2] = Nom du quiz
        # customdata[0] = Heure (HH:MM)
        # customdata[1] = Vraies minutes écoulées
        hovertemplate=(
            "<b>%{customdata[2]}</b><br>" +
            "Date: %{customdata[3]}<br>" +
            "Time: %{customdata[0]}<br>" +
            "Elapsed: %{customdata[1]} min<br>" +
            "Score: %{y:.2f}" +
            "<extra></extra>"
        )
    ))

    # 3. Class Average
    fig.add_trace(go.Scatter(
        x=plot_x,
        y=student_data["mean"],
        mode='markers',
        name="Class Avg",
        marker=dict(color="#e74c3c", symbol="x-thin",  line_width=2),
        hovertemplate="Class Avg: %{y:.2f}<extra></extra>"
    ))


    # 4. Zigzags
    for pos in pause_positions:
        # Vertical dashed line
        fig.add_vline(x=pos, line_dash="dash", line_color="#bdc3c7", line_width=1)
        
        # The Zigzag (using a small path at the bottom of the axis)
        # We draw a small 'Z' shape around y=0
        zz_width = 1  # width of the zigzag on the x-axis
        fig.add_shape(
            type="path",
            path=f"M {pos-zz_width},-0.05 L {pos+zz_width},0.05 M {pos-zz_width},0 L {pos+zz_width},0.1",
            line=dict(color="#7f8c8d", width=2),
            xref="x", yref="y"
        )
        
        fig.add_annotation(
            x=pos, y=0.6, text="LONG BREAK",
            showarrow=False, textangle=-90,
            font=dict(color="#95a5a6", size=10),
            bgcolor="white"
        )
    # 5. Final simplified Layout
    fig.update_layout(
        title=f"Timeline: {student_name}",
        xaxis=dict(
            title="Minutes from start",
            tickmode='array',
            tickvals=tick_vals,
            ticktext=tick_texts,
            #tickangle=0
        ),
        yaxis=dict(title="Score", range=[plot_y_min, plot_y_max]),
        template="plotly_white"
    )

    return fig

def natural_key(string_): #Gemini
    """Splits the string into a list of strings and integers."""
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', string_)]

def markdown_to_safe_html(text):
    text = text.replace("&lt;br&gt;", "<br>")
    # Conversion Markdown → HTML
    html_output = markdown.markdown(
        text,
        extensions=["extra", "sane_lists"]
    ).replace("<p>", "").replace("</p>", "")
    return html_output

def generate_pdf_report(html):
    from io import BytesIO
    from weasyprint import HTML, CSS

    buffer = BytesIO()
    HTML(string=html).write_pdf(target=buffer, stylesheets=[CSS(string=pdf_css)])
    pdf_bytes = buffer.getvalue()
    return pdf_bytes #HTML(string=html).write_pdf()

def prepare_student_data(df_last, marks_df, quiz_stats, selected_student):

    student_data = df_last[df_last["name"] + " " + df_last["firstname"] == selected_student].copy()
    student_data.drop(['send_timestamp', 'notebook_id', 'student', 'event_type', 'parameters', 'has_seen_correction'], axis=1, inplace=True)
    student_data['timestamp'] = pd.to_datetime(student_data['timestamp'].str.split(' \(').str[0])
    from_marks = marks_df[marks_df["full_names"] == selected_student]
    id_vars = ['name', 'firstname', 'class_group', 'FinalMark', 'full_names']
    df_marks = from_marks.melt(
        id_vars=id_vars, 
        var_name='quiz_title',   
        value_name='score'       
    ).drop(['name', 'firstname', 'class_group', 'full_names'], axis=1)

    student_data = student_data.merge(df_marks, on="quiz_title", suffixes=('_old', ''))
    student_data = student_data.merge(quiz_stats, on="quiz_title", suffixes=('_old', '')).sort_values("timestamp") 
    student_data["FinalMark"] = student_data["FinalMark"]*int(st.session_state.FinalMarkScale)/20

    student_data.index = student_data['quiz_title']

    return student_data

def make_individual_report(selected_student, df_last, student_data, quiz, final_weights, bareme, fullCorrection=True):
    
    FinalMarkScale = int(st.session_state.TrueFinalMarkScale)
    full_avg_note = st.session_state.df_final["FinalMark"].mean()*FinalMarkScale/20
    full_std_note = st.session_state.df_final["FinalMark"].std()*FinalMarkScale/20
    FinalMark = student_data.loc[:, 'FinalMark'].mean()*FinalMarkScale/20
    class_mean=full_avg_note
    class_std=full_std_note

    info_marks = _("Final mark: {FinalMark:.2f} / {FinalMarkScale} -- Class mean: {class_mean:.2f} & Standard deviation: {class_std:.2f}").format(FinalMark=FinalMark, 
                                        FinalMarkScale=FinalMarkScale, 
                                        class_mean=class_mean, class_std=class_std)


    html_output = True

    html = """
    <html>
    <head>
    <script>
    window.MathJax={tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']]}};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>

    <style>
    body{font-family:Arial,sans-serif;margin:40px}
    h1{border-bottom:2px solid #444;padding-bottom:5px}
    .indent{margin-left:20px}
    .question{margin-top:25px;padding:15px;border:1px solid #ddd;border-radius:8px;background:#fafafa}
    .prop{margin-left:10px;margin-top:4px}

    .correct,.incorrect,.missed{font-weight:bold}
    .correct{color:green}
    .incorrect{color:red}
    .missed{color:orange}

    .checkbox{font-family:monospace;margin-right:6px}

    .prop-row{display:flex;gap:10px;margin:4px 0 4px 20px}
    .col-checkbox{width:30px;font-family:monospace}
    .col-text{flex:1}
    .col-status{width:50px;font-weight:bold}
    .col-mark{width:50px;text-align:right}

    @page{size:A4;margin:10mm}
    </style>
    </head>
    <body>
        """

    block_size = 12
    n_cols = bareme.shape[1]
    bareme_html = ""
    for i in range(0, n_cols, block_size):
        sub_df = bareme.iloc[:, i:i+block_size]
        bareme_html += "<br>" + sub_df.to_html(index=True, float_format='{:.2f}'.format, classes='table table-bordered', border=1)

    if html_output:
        html += """<h1>{Report_for} {student_name}</h1>
            <p><b>{info_marks}</b></p>
            <p></p>
            <p>{scale_str} {scale}</p>
            """.format(Report_for=_("Correction for"), Student=(_("Student")),
                scale_str=_("Scale:"), scale=bareme_html, student_name=selected_student, info_marks=info_marks)
        html += '<hr style="border: none; border-top: 2px solid #000; margin: 20px 0;">'
    else:
        st.markdown(_("#### Report for ") + selected_student)
        st.write(info_marks)

    
    list_quizzes = sorted(df_last["quiz_title"].unique(), key=natural_key)
    for q in list_quizzes:
        current_quiz = quiz.quiz_bank[q]
        question = current_quiz['question']
        propositions = current_quiz['propositions']
        propositions.sort(key=lambda d: d["label"]) # sort on keys - mandatory for corrections
        constraints = current_quiz.get('constraints', {})
        quiz_type = current_quiz['type']

        correct_answers = {prop['label']: prop['expected'] for prop in current_quiz['propositions']}
        correct_answers = {k:correct_answers[k] for k in sorted(correct_answers)}
        try: 
            user_answers = student_data.loc[q, 'answers']
            user_answers = {k:user_answers[k] for k in sorted(user_answers)} # sort on keys (shoul be already sorted, but security)
        except:
            user_answers = {}
        
        score, total_possible, details = calculate_quiz_score(quiz_type, user_answers, propositions, 
                    question=question, weights=final_weights, constraints=constraints, return_details=True)

        question = details['question']
        propositions = details['propositions']
        marks = details['marks']
        violations = details['violations'] 

        if html_output:
            html += """
                <p></p> <h3 style="display: inline;">{q} - </h3>
                <span> <em>{question}</em> </span> 
                """.format(q=q, question=markdown_to_safe_html(question))
        else:
            st.markdown(f"## {q}")
            st.write("**{q}** -- {question}".format(q=q, question=question))


        if len(user_answers) == 0:
            if html_output:
                html += "<p class='indent'>{no_answer}</p>".format(no_answer=_("No answer"))
            else:
                st.write(_("No answer"))
            continue
        
        answererd_at = _("Answered at: ") + pd.to_datetime(student_data.loc[q, 'timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        if html_output:
            html += "<p class='indent'>{answererd_at}</p>".format(answererd_at=answererd_at)
        else:
            st.write(str(answererd_at))
    

        if 'mcq' in current_quiz['type']:
            if fullCorrection:
                for prop in propositions:
                    #checkbox = "☑" if user_answers.get(prop['label'], '') else "☐" 
                    checkbox = "✅" if user_answers.get(prop['label'], '') else "⬜"
                    Res = _("Correct") if user_answers.get(prop['label'], '') == prop['expected'] else _("Incorrect")
                    mark = float(marks.get(prop['label'], 0))    
                    if html_output:
                        html += """
                        <div class="prop-row">
                            <div class="col-checkbox">{checkbox}</div>
                            <div class="col-text">{propal}</div>
                            <div class="col-status">{Res}</div>
                            <div class="col-mark">{mark}</div>
                        </div>
                            """.format(checkbox=checkbox, propal=markdown_to_safe_html(prop['proposition']), Res=Res, mark=mark)
                    else:
                        st.write("{checkbox}  - {propal} - {Res} - ".format(checkbox=checkbox, propal=prop['proposition'], Res=Res) + _("Mark: ") + "{mark:.2f}".format(mark=mark)) 
            else:
                correct = sum([user_answers.get(prop['label'], '') == prop['expected'] for prop in propositions])
                incorrect = sum([user_answers.get(prop['label'], '') != prop['expected'] for prop in propositions])
                if html_output:
                    html += """
                    <div class="prop-row">
                        {correct_str} {correct} - {incorrect_str} {incorrect}</div>
                    </div>
                        """.format(correct=correct, correct_str=_("Correct"), incorrect=incorrect, incorrect_str=_("Incorrect"))
                else:
                    st.write("{correct} {correct_str} - {incorrect} {incorrect_str}".format(correct=correct, correct_str=_("Correct"), incorrect=incorrect, incorrect_str=_("Incorrect")))
        else:
            if fullCorrection:
                for prop in propositions:
                    pexpected = float(prop.get("expected", 0))
                    answer = user_answers.get(prop['label'], '')
                    mark = float(marks.get(prop['label'], 0))
                    #diff = abs(answer - pexpected)
                    #tol = max(prop.get("tolerance_abs", 0), 
                    #        prop.get("tolerance", 0.01) * abs(pexpected)) 
                    #Res = "Correct" if diff <= tol else "Incorrect"
                    Res = _("Incorrect") if mark == 0 else _("Correct")

                    if html_output:
                        html += """
                        <div class="prop-row indent">
                            <div class="col-checkbox">{checkbox}</div>
                            <div class="col-text">{propal}</div>
                            <div class="col-status">{Res}</div>
                            <div class="col-mark">{mark}</div>
                        </div>
                            """.format(checkbox=answer, propal=markdown_to_safe_html(prop['proposition']), Res=Res, mark=mark)
                    else:
                        st.write("Given answer: {answer}  - {propal} - {Res} - ".format(checkbox=checkbox, propal=prop['proposition'], Res=Res) + _("Mark: ") + "{mark:.2f}".format(mark=mark)) 
            else:
                correct = sum([user_answers.get(prop['label'], '') == prop['expected'] for prop in propositions])
                incorrect = sum([user_answers.get(prop['label'], '') != prop['expected'] for prop in propositions])
                if html_output:
                    html += """
                    <div class="prop-row indent">
                        {correct_str} {correct} - {incorrect_str} {incorrect}</div>
                        <div class="col-mark"></div>
                    </div>
                        """.format(correct=correct, correct_str=_("Correct"), incorrect=incorrect, incorrect_str=_("Incorrect"))
                else:
                    st.write("{correct_str} {correct} - {incorrect_str} {incorrect}".format(correct=correct, correct_str=_("Correct"), incorrect=incorrect, incorrect_str=_("Incorrect")))
        
        if len(violations) > 0:
            logical_violations = _("Logical Contraints Violations: ")
            if html_output:
                html += "<h3>" + logical_violations + "</h3>"
                html += "<div class='indent'>"
            else:
                st.write(logical_violations)

            for key in violations:
                if fullCorrection:
                    current_violation = _("{key}: Violation between {indexes} - Malus:{malus}").format(key=key, 
                                                    indexes=violations[key]['indexes'], malus=violations[key]['malus'] )
                    if html_output:
                        html += current_violation
                    else:
                        st.write(current_violation)
                else:
                    current_violation = _("Violation {key}: Malus:{malus}").format(key=key, 
                                                    malus=violations[key]['malus'] )
                    if html_output:
                        html += current_violation
                    else:
                        st.write(current_violation)

            if html_output:
                html += "</div>"

        # Final score
        score_obtained = "Score obtained:  {score} / {total_possible} = {normalized_score:.2f} (class average: {class_avg:.2f} with std: {class_std:.2f})".format(
                    score=score, total_possible=total_possible, normalized_score=score/total_possible,
                    class_std=student_data.loc[q, 'std'], class_avg=student_data.loc[q, 'mean'] )
        if html_output:
            html += "<br>" + score_obtained 
            html += """
                    </body>
                    </html>"""
        else:
            st.write(score_obtained) 


    return html


def generate_zip_report(students_list, df_last, marks_df, quiz_stats, quiz, 
                        final_weights, fullCorrection, progress_callback=None, pdf_output=True):
    total = len(students_list)
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for k, student in enumerate(students_list):
            student_data = prepare_student_data(df_last, marks_df, quiz_stats, student)
            html_bytes = make_individual_report(student, df_last, student_data, quiz, 
                    final_weights, st.session_state.scale, fullCorrection=fullCorrection)
            if pdf_output:
                pdf_bytes = generate_pdf_report(html_bytes)
                zip_file.writestr(student + ".pdf", pdf_bytes)
            else:
                zip_file.writestr(student + ".html", html_bytes)
            if progress_callback:
                progress_callback((k + 1) / total)
    zip_buffer.seek(0)
    zip_bytes = zip_buffer.getvalue()
    return zip_bytes
    

# CSS for PDF exports
pdf_css = """
@page {
    size:A4;
    page-size: A4;
    margin: 2.5cm;
    /* Footer area for numbering */
    @bottom-right {
        content: "Page " counter(page) "/" counter(pages);
        font-family: "Liberation Serif", serif;
        font-size: 9pt;
        color: #555;
    }
}

body {
    font-family: "Liberation Serif", "Times New Roman", serif;
    font-size: 11pt;
    line-height: 1.4;
    color: black;
}

/* Title hierarchy */
h1 {
    font-size: 14pt;
    font-weight: bold;
    text-transform: uppercase;
    margin-bottom: 0.5cm;
}

h2 {
    font-size: 12pt;
    font-weight: bold;
    margin-top: 0.4cm;
    margin-bottom: 0.2cm;
    border-bottom: 0.5pt solid #ccc; /* Small discreet line to separate sections */
}

h3 {
    font-size: 11pt; /* Same size as body, but bold/italic */
    font-weight: bold;
    font-style: italic;
    margin-top: 0.3cm;
    margin-bottom: 0.1cm;
}

/* Handling intelligent page breaks */
h1, h2, h3 {
    page-break-after: avoid; /* Avoid a title being alone at the bottom of the page */
}
"""


#-------------------------------------------------
#                      MAIN                      #
#-------------------------------------------------
def main():
    import streamlit as st
    global _
    global local_storage #
    
    # Unique identifier for "tabs"
    if "render_id" not in st.session_state:
        st.session_state.render_id = 0
    st.session_state.render_id += 1

    # This is to stabilize the iframe, in the rendering flow
    storage_container = st.empty()
    with storage_container:
        local_storage = LocalStorage()
    

    old_setItem = local_storage.setItem
    #local_storage.setItem = lambda key, value: old_setItem(key, json.dumps(value), key=key)
    local_storage.setItem = lambda key, value: old_setItem(key, value, key=key+'_'+str(time.time()))
    old_deleteItem = local_storage.deleteItem
    local_storage.deleteItem = lambda key: old_deleteItem(key, key=key+'_'+str(time.time()))

    monitored_parameters = ["selected_lang", "url", "secret", "params_str", "maxtries",
                            "groups", "group","seuil", "exam_title",  "bareme_str",
                            "main_nav_state", "monitoring_nav_state", "correction_nav_state", 'FinalMarkScale']
    

    if "_init" not in st.session_state:
        # Default values (replace value=...)
        set_defaults()

        # Restore the session state
        for k in monitored_parameters:
            stored = local_storage.getItem(k)
            if verbose: print(k, stored)
            if stored:
                st.session_state[k] = json.loads(stored)
                if verbose: print("Restored", k, st.session_state[k])
                pass
        st.session_state["_init"] = True

    
    # --- 3. DYNAMIC FILE RESTORATION LOGIC ---
    # We check whether the widget is empty BUT we have a backup
    
    package_json = local_storage.getItem("file_package")
    #print("file_package", package_json)
    #package_json = False

    if st.session_state.get("quiz_file") is None and package_json:
        # If the restored object is not already in memory, we recreate it
            if "restored_file" not in st.session_state:
                try:
                    package = json.loads(package_json)
                    # On extrait les deux infos du dictionnaire unique
                    raw_bytes = zlib.decompress(base64.b64decode(package["b64"]))
                
                    restored_file = io.BytesIO(raw_bytes)
                    restored_file.name = package["name"] # Le nom est ici !
                    restored_file.size = len(raw_bytes)
                    restored_file.seek(0)
                    st.session_state["restored_file"] = restored_file
                except Exception as e:
                        st.error(f"Restoration error: {e}")
    else:
        # If the widget is filled, we make sure not to use the old restored file
        st.session_state.pop("restored_file", None)
    # -------- End persistence --------------


    # Language selection
    #lang = st.sidebar.selectbox("Language", ["🇬🇧 en", "🇫🇷 fr"], index=["en", "fr"].index(st.session_state.lang))
    languages = {
        "en": "🇬🇧 English",
        "fr": "🇫🇷 Français",
        "es": "🇪🇸 Spanish",
    }

    lang = st.sidebar.selectbox(
        "Language",
        options=list(languages.keys()),
        format_func=lambda x: languages[x],
        index=list(languages.keys()).index(st.session_state.lang),
        key = "selected_lang",
        on_change=sync, args=("selected_lang",)
    )

    if lang != st.session_state.lang:
        print(f"Language changed from {st.session_state.lang} to {lang}")
        _ = set_language(lang)
        #st.rerun()

    

    # --- PAGE CONFIGURATION ---
    st.set_page_config(page_title=_("Dashboard LabQuiz"), layout="wide", 
                    page_icon="src/quiz_dash/1F4CA.png")#"📊")
    
    # Custom styles
    apply_custom_styles()

    parameters_placeholder = st.container()
    group_placeholder = st.container()
    #tabs_placeholder = st.empty()


    # --- SIDEBAR: CONNECTION AND REFRESH RATE ---
    with st.sidebar:
        st.header(_("🔑 Connection"))
        url = st.text_input(_("Google Sheet URL"), placeholder="https://docs.google.com/...", 
                            key="url", on_change=sync, args=("url",))
        secret = st.text_input(_("Secret Key"), type="password", key="secret",
                               on_change=sync, args=("secret",))
        label =_('QUIZ file (YAML) containing corrections')
        st.markdown(
            f"<span style='font-size:0.8rem; color: black;'>{label}</span>",
             unsafe_allow_html=True
            )
        #st.caption(_("QUIZ file (YAML) containing corrections"))
        dynamic_key = f"quiz_file_{st.session_state.uploader_version}"
        uploaded_file = st.file_uploader("label", type=["yaml"], key=dynamic_key, 
                                     label_visibility="collapsed",
                                     #on_change=mark_upload, args=("quiz_file",)
                                     )
        
        if uploaded_file is not None:
            current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.last_processed_file != current_file_id:
                for key in st.session_state.keys():
                    if key.startswith("quiz_file"):
                        key_quiz = key
                        break
                sync(key_quiz)
                st.session_state.last_processed_file = current_file_id
        elif uploaded_file is None and st.session_state.last_processed_file is not None:
            # Case where user deletes file manually (red cross)
            st.session_state.last_processed_file = None
       

        quiz_file = uploaded_file if uploaded_file is not None else st.session_state.get("restored_file")
        st.divider()
    
        # Group selection (if several groups)
        if "groups" in st.session_state:
            if verbose: print("groups exist")
            st.markdown(f"### 👥 {_('Class/Group selection')}")
            group = st.selectbox(
                _("Class/Group"), 
                st.session_state.groups, 
                key="group", 
                on_change=sync, 
                args=("group",)
            )
            st.divider()
        else:
            if verbose: print("groups does not exist")

        
        #st.header(_("⏱️ Monitoring"))
        st.markdown("### ⏱️ " + _("Monitoring"))
        refresh_count = -1

        refresh_min = st.slider(_("Refresh rate (min)"), 1, 30, 10)
        auto_refresh_active = st.checkbox(_("Enable auto-refresh"), value=False)

        if st.button(_("🔄 Refresh now"), use_container_width=True):
            st.session_state.refresh_key += 1

        if auto_refresh_active:
            st.caption(_('Last update: ') + time.strftime('%H:%M:%S'))
            refresh_count = st_autorefresh(
                interval=refresh_min * 60 * 1000,
                key="datarefresh"
            )

        st.divider()

        # Button Reset
        if st.button("🗑️ Global reset", use_container_width=True, on_click=perform_global_reset):
            pass
            #st.session_state["_reset_app"] = True
            #st.rerun()


    # --- MAIN AREA: SETTINGS ---
    with parameters_placeholder:
        st.title(_("📊 Monitoring & Correction Dashboard"))

        with st.expander(_("🛠️ Parameter Configuration (Integrity & Correction)"), expanded=True):
            col_p1, col_p2 = st.columns(2)
            
            with col_p1:
                st.markdown(_("**Monitoring & Source**"))
                params_str = st.text_input(_("Parameters to monitor (e.g.: {'retries':2, 'exam_mode':False, 'test_mode':False})"), 
                                        #value="{'retries':2, 'exam_mode':False, 'test_mode':False}", 
                                        key="params_str", on_change=sync, args=("params_str",))
                maxtries = st.number_input(_("Number of allowed attempts"), min_value=1, 
                                        #value=3, 
                                        key="maxtries", on_change=sync, args=("maxtries",))
                
            with col_p2:
                st.markdown(_("**Grading Algorithm**"))
                seuil = st.number_input(_("Threshold (0 to avoid negative marks)"), 
                                        #value=0.0, 
                                        key="seuil", on_change=sync, args=("seuil",))
                exam_title = st.text_input(_("Exam title (if randomized)"), 
                                        #value="", 
                                        key="exam_title", on_change=sync, args=("exam_title",))

            st.divider()
            col_p3, col_p4 = st.columns(2)
            
            with col_p3:
                st.markdown(_("**Weights Matrix**"))
                # Dictionary editor for base weights
                weights_dict = st.data_editor({
                    _("TP (True Positive)"): 1.0,
                    _("FP (False Positive)"): -1.0,
                    _("FN (False Negative)"): 0.0,
                    _("TN (True Negative)"): 0.0
                }, key="weights_editor",  args=("weights_editor",))
                
                # Conversion for correctQuizzesDf function
                final_weights = {
                    (True, True): weights_dict[_("TP (True Positive)")],
                    (True, False): weights_dict[_("FP (False Positive)")],
                    (False, True): weights_dict[_("FN (False Negative)")],
                    (False, False): weights_dict[_("TN (True Negative)")]
                }

            with col_p4:
                st.markdown(_("**Grading scale per question**"))
                bareme_str = st.text_area(_("Scale dictionary (e.g.: {'q1': 2})"), 
                                        #value="{}", 
                                        key="bareme_str", on_change=sync, args=("bareme_str",)) #key="bareme_str")


    # --- DATA PROCESSING ---

    if url and secret and quiz_file:
        try:
            import copy
            # 1. Reading
            read_error = False
            with st.spinner(_("Reading data...")):
                full_df, full_df_filt = adhocReadData(url, secret, refresh_count, st.session_state.refresh_key)
                if full_df is None or full_df_filt is None:
                    st.error(_("Data could not be read."))
                    read_error = True
            if read_error: st.stop()
            full_df = generate_cols_from_student(full_df, dropStudent=False)
            full_df_filt = generate_cols_from_student(full_df_filt, dropStudent=False)
            # Group selection
            group = _('All')

            if 'class_group' in full_df.columns:
                all_groups = [
                    g for g in full_df['class_group'].unique()
                    if g is not None and not pd.isnull(g)
                ]
                all_groups.sort()
                if all_groups:
                    #st.info(_("Select a class or group to monitor."))
                    groups = [ _('All') ] + all_groups
                    st.session_state.groups = groups
                    sync("groups")
            
                
            # Filtering
            if "group" in st.session_state:
                group = st.session_state.group
                if verbose: print("Current group from state:", group)
            else:
                if verbose: print("Group key is not in session state yet")


            if group == _('All'):
                df, df_filt = full_df, full_df_filt
            else:
                df = full_df.query("class_group == @group")
                df_filt = full_df_filt.query("class_group == @group")                


            # 2. Instantiate a quiz with the quiz file CONTAINING expected values
            
            from labquiz.utils import get_full_object_hash, get_big_integrity_hash
            params = eval(params_str)
            quiz = QuizLab("", quiz_file, needAuthentification=False, mandatoryInternet=False, 
                        in_streamlit=True, silentStart=True, **params)              
            wanted_hash = get_full_object_hash(quiz, modules=['main', 'utils'], 
                                                WATCHLIST=['retries', 'exam_mode', 'test_mode'])

            # 3. Global integrity check (hash)
            # wanted_hash = # To be defined! st.secrets["hash"]
            # check_hash_integrity(df, 'full', wanted_hash=wanted_hash) # Will display in terminal or via st if modified
            
            # 4. Students retrieval
            students_raw = sorted(list(df["student"].dropna().unique()))
            students = [s for s in students_raw]
            if "render_id" not in st.session_state:
                st.session_state.render_id = 0
            st.session_state.render_id += 1

            tabs_placeholder = st.empty()
            with tabs_placeholder.container(border=True, key=f"main_frame_{st.session_state.render_id}"): 
                st.empty()
                st.markdown(f"### 🛠️ {_('Live monitoring & Correction')}")
                tab_names = [_("📡 Integrity Live"), _("👀 Monitoring"), _("🎯 Correction & Grades")]
                #selected_tab = st.radio(_("Select a tab"), 
                #                        tab_names, horizontal=True, 
                #                        label_visibility="collapsed",
                #                        key="active_tab")
                #tab_mon, tab_mon_graph, tab_corr = st.tabs(tab_names)
                selected_tab = st.segmented_control(
                    label="Navigation",
                    options=tab_names,
                    key="main_nav_state", 
                    label_visibility="collapsed",
                    on_change=sync, 
                    args=("main_nav_state",),
                )
                st.divider() 
                #with tab_mon:
                if selected_tab == _("📡 Integrity Live"):
                    from labquiz.putils import make_anomalies_df_report, group_anomalies_per_student

                    st.subheader(_("Real-time integrity monitoring"))
                    monitoring_data = [] 
                    
                    # Secure parameters evaluation
                    try:
                        reference = eval(params_str)
                    except Exception as e:
                        p_list = {}
                        st.error(_("Error in parameters format."))
                        print(e)

                    includeRAS = True
                    if st.checkbox(_("Also use full hash"), value=False, 
                                help=_("Use the full hash of the source code, live object and parameters")):
                        reference['full_hash'] = wanted_hash
                    if st.checkbox(_("Only display anomalies"), value=False, 
                                help=_("Display anomalies only, or full report")):
                        includeRAS = False

                    Tab_report = make_anomalies_df_report(df, reference, ignore_keys=[], 
                                                        includeRAS=includeRAS)

                    if st.checkbox(_("Collect anomalies per student"), value=False, 
                                help=_("Group anomalies per student")):
                        Grouped_tab_report = group_anomalies_per_student(Tab_report)
                        if not Grouped_tab_report.empty:
                            st.dataframe(Grouped_tab_report, width='stretch', hide_index=True)
                        else:
                            st.info(_("No anomalies found at all."))
                    else:
                        st.dataframe(Tab_report, width='stretch', hide_index=True)

                
                ## New Monitoring tab
                elif selected_tab == _("👀 Monitoring"): 
                    st.subheader(_("Activity monitoring"))
                    
                    # 1. Data Preparation
                    df_last = prepare_monitoring_data(df)

                    if df_last.empty:
                        st.info(_("No valid activity recorded yet."))
                    else:
                        monitoring_tab_names = [_("📊 Monitoring charts"), _("🕵️‍♀️ Activity Summary"), _('Student Timeline')]
                        monitoring_tab = st.segmented_control(
                            label="Navigation",
                            options=monitoring_tab_names,
                            default=monitoring_tab_names[0],
                            key="monitoring_nav_state", 
                            label_visibility="collapsed",
                            on_change=sync, 
                            args=("monitoring_nav_state",),
                        )

                        if monitoring_tab == monitoring_tab_names[0]:
                            # 2. Grid Layout (2 columns)
                            # Row 1
                            col1, col2 = st.columns(2)
                            with col1:
                                fig_counts = create_monitoring_plot(df_last, _("Quizzes Completed per Student"), "student_counts", _)
                                st.plotly_chart(fig_counts, use_container_width=True)
                                #st.plotly_chart(fig, use_container_width=True)
                            with col2:
                                fig_scores = create_monitoring_plot(df_last, _("Total Scores per Student"), "student_scores", _)
                                st.plotly_chart(fig_scores, use_container_width=True)

                            # Row 2
                            col3, col4 = st.columns(2)
                            with col3:
                                fig_class = create_monitoring_plot(df_last, _("Class Progress per Quiz"), "class_results", _)
                                st.plotly_chart(fig_class, use_container_width=True)
                            with col4:
                                #fig_hardest = create_monitoring_plot(df_last, _("Top 5 Hardest Quizzes (Avg Score)"), "hardest_quizzes", _)
                                #st.pyplot(fig_hardest, width="stretch")
                                fig_most_selective = create_monitoring_plot(df_last, _("Quizzes Selectivity(Avg Score)"), "quizzes_selectivity", _)
                                st.plotly_chart(fig_most_selective, use_container_width=True)
                                    
                        elif monitoring_tab == monitoring_tab_names[1]:
                        

                            # 3. Detailed Data Table (below the grid)
                            st.markdown(_("#### Detailed Activity Summary"))
                            
                            # Preparing the table data
                            detailed_stats = (
                                df_last.groupby("student")
                                .agg(
                                    nb_quizzes=("quiz_title", "size"),
                                    total_score=("score", "sum"),
                                    quizzes_list=("quiz_title", lambda x: ", ".join(list(x)))
                                )
                                .reset_index()
                            )
                            
                            detailed_stats['student'] = detailed_stats['student'].apply(
                                lambda x: x.split(',')[0].strip().upper() + ' ' + x.split(',')[1].strip().title() 
                                    if len(x.split(',')) > 1 else x
                                    )

                            # Displaying the dataframe with formatted headers
                            st.dataframe(
                                detailed_stats,
                                column_config={
                                    "student": st.column_config.TextColumn(_("Student")),
                                    "nb_quizzes": st.column_config.NumberColumn(_("Count")),
                                    "total_score": st.column_config.NumberColumn(_("Total Points"), format="%.1f"),
                                    "quizzes_list": st.column_config.TextColumn(_("List of Quizzes"))
                                },
                                hide_index=True,
                                width='stretch'
                            )

                        elif monitoring_tab == monitoring_tab_names[2]:
                            # 4. Student Timeline
                            st.markdown(_("#### Student Timeline"))
                            
                            # 1. Global stats calculation
                            quiz_stats = df_last.groupby("quiz_title")["score"].agg(["mean", "std"]).reset_index()

                            # 2. Student Selection
                            all_students = sorted(df_last["student"].unique())
                            selected_student = st.selectbox("Select a student", all_students)

                            # 3. Display Plot
                            if selected_student:
                                # 1. Data Prep
                                student_data = df_last[df_last["student"] == selected_student].copy()
                                student_data['timestamp'] = pd.to_datetime(student_data['timestamp'].str.split(' \(').str[0])
                                student_data = student_data.merge(quiz_stats, on="quiz_title").sort_values("timestamp")                          
                                # 2. Plot
                                fig_timeline = plot_student_session_track(student_data, selected_student)
                                st.plotly_chart(fig_timeline, use_container_width=True)


                ## End of new monitoring tab
                elif selected_tab == _("🎯 Correction & Grades"):
                    
                    st.subheader(_("Correction, Grades & Reports"))

                    correction_tab_names = [_("🎯 Correction & Grades"), _('✍🏻 Students reports')]
                    correction_tab = st.segmented_control(
                        label="Navigation",
                        options=correction_tab_names,
                        key="correction_nav_state", 
                        label_visibility="collapsed",
                        on_change=sync, 
                        args=("correction_nav_state",),
                    )

                    if correction_tab == correction_tab_names[0]:
                    #with tab_corr:
                        col_res1, col_res2 = st.columns([99, 1])
                        b_dict = {}
                        with col_res1:    
                            st.caption(_('Last update of corrections (before auto-refresh): ') + f"{st.session_state.last_correction_update}")
                            if st.button(_("🚀 Launch full correction"), width='stretch'):
                                st.session_state.last_correction_update = time.strftime('%H:%M:%S')
                                st.caption(_('Last update of corrections: ') + f"{st.session_state.last_correction_update}")
                                with st.spinner(_("Calculating scores per question...")):
                                    try:
                                        b_dict = eval(bareme_str)              
                                    except:
                                        b_dict = {}
                                    
                                    st.session_state.df_results = correctQuizzesDf(
                                        data=full_df, 
                                        data_filt=full_df_filt, 
                                        quiz=quiz, 
                                        title=exam_title if exam_title != "" else None, 
                                        seuil=seuil, 
                                        weights=final_weights, 
                                        bareme=b_dict, 
                                        maxtries=maxtries
                                    )
                                    st.session_state.df_results = st.session_state.df_results.reset_index().rename(columns={"index": "student"})
                                    st.session_state.df_results = generate_cols_from_student(st.session_state.df_results, dropStudent=True)
                                    st.session_state.df_results.drop(columns='Note', inplace=True, errors='ignore')
                                    st.session_state.df_results.drop(columns='maxpts', inplace=True, errors='ignore')
                                    
                                    if exam_title == "":
                                        st.session_state.df_results.drop(columns='Note', inplace=True, errors='ignore')
                                    st.success(_("Scores calculated!"))
                            
                                    questions = [c for c in st.session_state.df_results.columns if c not in [
                                        "student", "maxpts", "Note", "FinalMark", 'name', 'firstname', 'class_group']
                                        ]                    
                                    st.session_state.coeffs = {q: float(b_dict.get(q, 1.0)) for q in questions}  
                        
                                st.session_state.df_final = st.session_state.df_results.copy() # All groups in df_final
                            
                                
                            if st.session_state.df_final is not None:

                                if group != _("All"):
                                    st.session_state.df_results = st.session_state.df_final.query("class_group == @group")
                                else:
                                    st.session_state.df_results = st.session_state.df_final
                                st.session_state.show_scores = True
                                    
                                st.markdown(_("#### ⚖️ Adjust Scale"))
                                questions = [c for c in st.session_state.df_results.columns if c not in ["student", "maxpts", "Note", "FinalMark", 'name', 'firstname', 'class_group']]
                                
                                # Create a mini-table to adjust weights without recalculating everything
                                adj_bareme = st.data_editor(
                                    pd.DataFrame({ "AvgScore": st.session_state.df_results[questions].mean(axis=0), 
                                                "Coefficient": st.session_state.coeffs.values()},
                                                ).transpose(),
                                    hide_index=False,
                                    width='stretch',
                                )
                                st.session_state.scale = adj_bareme

                                st.session_state.df_results = recompute_score(
                                    adj_bareme, 
                                    questions, 
                                    exam_title, 
                                    st.session_state.df_results, 
                                    full_df, 
                                    full_df_filt, 
                                    quiz, 
                                    seuil, 
                                    final_weights, 
                                    maxtries
                                )
                                st.session_state.show_scores = True

                                col1, col2, col3 = st.columns([4, 1, 1], vertical_alignment="bottom")
                                with col1:
                                    st.markdown(_("#### Grades Table"))
                                with col2:
                                    if "TrueFinalMarkScale" in st.session_state:
                                        # Seems that FinalMarkScale is overwritten by browser cache sometimes. This is a Fallback
                                        st.session_state.FinalMarkScale = st.session_state.TrueFinalMarkScale

                                    options = ["100", "20", "4"]
                                    current_index = options.index(str(st.session_state.FinalMarkScale)) if str(st.session_state.FinalMarkScale) in options or st.session_state.FinalMarkScale in options else 0
                                    FinalMarkScale = st.selectbox(_("Final Mark Scale:"), options, 
                                            key="FinalMarkScale", on_change=sync, args=("FinalMarkScale",))

                                avg_note = st.session_state.df_results["FinalMark"].mean()*int(FinalMarkScale)/20
                                std_note = st.session_state.df_results["FinalMark"].std()*int(FinalMarkScale)/20
                                st.caption(_('Average: ') + f"{avg_note:.2f} / {FinalMarkScale}. " + _('Standard deviation: ') + f"{std_note:.2f}")
                                # Compute stats for All groups
                                coeffs = adj_bareme.loc["Coefficient"]
                                st.session_state.df_final["FinalMark"] = st.session_state.df_final[questions].dot(coeffs)*(20/sum(coeffs))
                                full_avg_note = st.session_state.df_final["FinalMark"].mean()*int(FinalMarkScale)/20
                                full_std_note = st.session_state.df_final["FinalMark"].std()*int(FinalMarkScale)/20
                                
                                if group != _("All"):
                                    st.caption(_('Class: ') + _('Average: ') + f"{full_avg_note:.2f} / {FinalMarkScale}. " + _('Standard deviation: ') + f"{full_std_note:.2f}")
                                display_scores = st.session_state.df_results.copy()
                                display_scores["FinalMark"] = display_scores["FinalMark"]*int(FinalMarkScale)/20
                                
                                with col3:
                                    st.button(_("Histogram"), key="histogram", on_click=show_histogram, args=(display_scores["FinalMark"],) )
                                        #show_histogram(display_scores["FinalMark"])
                                
                                st.dataframe(display_scores, width='stretch')
                                
                                st.session_state.show_scores = False 
                            elif st.session_state.df_results is not None:
                                st.info(_("Scores per question available. Click calculate to see final marks."))
                            else:
                                st.caption(_("Waiting for correction to start."))
                    
                    elif correction_tab == correction_tab_names[1]:
                        if st.session_state.df_final is not None:

                            df_last = prepare_monitoring_data(full_df_filt)
                            
                            # 1. Global stats calculation
                            quiz_stats = df_last.groupby("quiz_title")["score"].agg(["mean", "std"]).reset_index()
                            marks_df = st.session_state.df_final.copy()

                            with st.expander(_("🎓 Individual analysis"), expanded=True):
                                # 2. Student Selection
                                marks_df['full_names'] = marks_df["name"] + " " + marks_df["firstname"]
                                all_students = sorted(marks_df['full_names'].unique())
                                st.session_state.all_students = all_students
                                selected_student = st.selectbox("Select a student", all_students)

                                if selected_student:
                                    with st.expander(_("🛣️ Student Timeline"), expanded=False):
                                    #st.markdown(_("#### Student Timeline"))
                                    
                                    # 3. Display Plot
                                        # 1. Data Prep
                                        student_data = prepare_student_data(df_last, marks_df, quiz_stats, selected_student)

                                        st.dataframe(student_data, width='stretch')   

                                        # 2. Plot
                                        fig_timeline = plot_student_session_track(student_data, selected_student)
                                        st.plotly_chart(fig_timeline, use_container_width=True)

                                    
                                    
                                    # 4. Make Report
                                    with st.expander(_("📜 Individual Report"), expanded=False):
                                        
                                        was = '''FinalMarkScale = int(st.session_state.FinalMarkScale)
                                        full_avg_note = st.session_state.df_final["FinalMark"].mean()*FinalMarkScale/20
                                        full_std_note = st.session_state.df_final["FinalMark"].std()*FinalMarkScale/20
                                        FinalMark = student_data.loc[:, 'FinalMark'].mean()
                                        class_mean=full_avg_note
                                        class_std=full_std_note

                                        info_marks = _("Final mark: {FinalMark:.2f} / {FinalMarkScale} -- Class mean: {class_mean:.2f} & Standard deviation: {class_std:.2f}").format(FinalMark=FinalMark, 
                                                                            FinalMarkScale=FinalMarkScale, 
                                                                            class_mean=class_mean, class_std=class_std)'''
                                        
                                        col1, col2, col3 = st.columns([4, 4, 4], vertical_alignment="bottom")
                                        with col3:
                                            fullCorrectionInReport = st.checkbox(
                                                _("Full correction in Report"),
                                                value=True,
                                                key="fullCorrectionInReport",
                                            )

                                        html = make_individual_report(selected_student, df_last, student_data, quiz, final_weights, st.session_state.scale,  fullCorrection=fullCorrectionInReport)

                                        with col1:
                                            st.download_button(
                                                    label=_("Download PDF report"),
                                                    data=generate_pdf_report(html),
                                                    file_name=f"{selected_student}.pdf",
                                                    mime="application/pdf"
                                                )
                                        with col2:
                                            st.download_button(
                                                    label=_("Download HTML report"),
                                                    data=html,
                                                    file_name=f"{selected_student}.html",
                                                    mime="application/pdf"
                                                )
                                    
                                            
                                        st.markdown("---")

                                        #st.markdown(html, unsafe_allow_html=True)  
                                        components.html(html, height=1000, scrolling=True)  


                            # 5. Download Full Reports
                            with st.expander(_("📔 All Class Reports"), expanded=False):
                                col1, col2, col3 = st.columns([4, 4, 4])#, vertical_alignment="bottom")
                                with col3:
                                    fullCorrectionInAllReports = st.checkbox(
                                        _("Full correction in Report"),
                                        value=True,
                                        key="fullCorrectionInAllReports",
                                    )

                                if "zipped_pdf_reports" not in st.session_state:
                                    st.session_state.zipped_pdf_reports = None

                                if "zipped_html_reports" not in st.session_state:
                                    st.session_state.zipped_html_reports = None

                                with col1:
                                    if st.button(_("Generate zip of all PDF reports")):
                                        progress_bar = st.progress(0)
                                        def update_progress(p):
                                            progress_bar.progress(p)

                                        with st.spinner(_("Generating zip...")):
                                            st.session_state.zipped_pdf_reports = generate_zip_report(
                                                all_students, df_last, marks_df, quiz_stats, quiz, 
                                                final_weights, st.session_state.fullCorrectionInAllReports, 
                                                progress_callback=update_progress, pdf_output=True)

                                    if st.session_state.zipped_pdf_reports is not None:
                                        st.download_button(
                                                label=_("Download zip of all PDF reports"),
                                                data=st.session_state.zipped_pdf_reports,
                                                file_name="all_pdf_reports.zip",
                                                mime="application/zip"
                                            )
                                with col2:
                                    if st.button(_("Generate zip of all HTML reports")):
                                        progress_bar = st.progress(0)
                                        def update_progress(p):
                                            progress_bar.progress(p)

                                        with st.spinner(_("Generating zip...")):
                                            st.session_state.zipped_html_reports = generate_zip_report(
                                                all_students, df_last, marks_df, quiz_stats, quiz, 
                                                final_weights, st.session_state.fullCorrectionInAllReports, 
                                                progress_callback=update_progress,  pdf_output=False)
                                            
                                    if st.session_state.zipped_html_reports is not None:
                                        st.download_button(
                                                label=_("Download zip of all HTML reports"),
                                                data=st.session_state.zipped_html_reports,
                                                file_name="all_html_reports.zip",
                                                mime="application/zip"
                                            )

                        else:
                            st.info(_("Full correction must be computed first (see Correction & Grades tab)."))


        except Exception as e:
            st.error(_('Error during reading or processing: ') + f"{e}")
    else:
        st.warning(_("Please enter URL and SECRET in the sidebar, and load a quiz file."))

if __name__ == "__main__":
    main()
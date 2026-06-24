---
title: LabQuiz overview
subtitle: 'Interactive Quizzes Inside Jupyter Notebooks — with Editor & Live Dashboard'
date: 2026-03-25
license: CC-BY-NC-SA-4.0
settings:
    myst_to_tex:
        code_style: listings
    output_stderr: remove
    output_matplotlib_strings: remove
exports:
  - format: docx
  - format: pdf
    template: arxiv_nips
    article_type: article
    chapters: []
---


**labquiz** is a Python package that allows you to seamlessly integrate interactive quizzes directly into Jupyter notebooks — useful for labs, tutorials, practical assignments, continuous assessment, and controlled exams.

It combines:

* ✅ Multiple-choice and numerical questions
* 🧩 Template-based parameterized questions
* 🔁 Configurable number of attempts
* 💡 Hints and detailed feedback
* 📊 Automatic scoring
* 🌐 Optional remote logging (Google Sheets)
* 📈 Real-time monitoring dashboard (if logging)
* 🔐 Integrity checks and anti-tampering mechanisms

And it comes with two optional companion tools:

* ✏️ **`quiz_editor`** — Create, edit, encrypt, and export question banks
* 📊 **`quiz_dash`** — Monitor, correct, and analyze results in real time

Together, these components make up the **LabQuiz** ecosystem, which is distributed as a bundle, `labquizbundle`. 

---

```mermaid
flowchart LR
    %% Nodes
    A["YAML Quiz<br/>(Quiz-as-Code)"]
    B["quiz_editor<br/>Authoring Interface<br/>(optional)"]
    C["Jupyter Notebook<br/>+ Interactive Quizzes"]
    D["Remote Backend<br/>(e.g., Google Sheets)"]
    E["quiz_dash<br/>Monitoring Dashboard"]
    F[Analytics]
    G[Marks Table]
    H[Student Reports]
    I[Exports: <br/>Web-based Training Quizzes <br/>AMC / LaTeX Paper Quizzes <br/> Moodle]

    %% Flows
    B -.-> A
    A ---> C
    C -.-> D
    D --> E
    E --> F
    E --> G
    E --> H
    A -.-> I
    

    %% Styles
    classDef authoring fill:#e3f2fd,stroke:#1e88e5,stroke-width:1px;
    classDef runtime fill:#e8f5e9,stroke:#43a047,stroke-width:1px;
    classDef backend fill:#fff3e0,stroke:#fb8c00,stroke-width:1px;
    classDef dashboard fill:#f3e5f5,stroke:#8e24aa,stroke-width:1px;
    classDef outputs fill:#fce4ec,stroke:#d81b60,stroke-width:1px;
    classDef export fill:#e0f7fa,stroke:#00838f,stroke-width:1px;

    %% Assign classes
    class A,B authoring;
    class C runtime;
    class D backend;
    class E dashboard;
    class F,G,H outputs;
    class I,J export;
```
Figure: Overview of the `LabQuiz` ecosystem. Dashed arrows indicate optional components. YAML quizzes drive both interactive notebooks and exports, monitored via a dashboard producing analytics, marks tables, and student reports.

---

* 👉🏼 `Live version`  Try it in [binder](https://mybinder.org/v2/gh/jfbercher/labquiz/main?urlpath=%2Fdoc%2Ftree%2Fextras%2FlabQuizDemo_en_binder.ipynb) 
* `Installation`: 
  All packages can be installed at once (`labquiz`, `quiz_editor` and `quiz_dash`)
```bash
# From PyPI
   pip install labquizbundle
```
or individually, e.g.
```bash
   pip install labquiz
```
They can also be installed from source e.g.
```bash
# From source
pip install git+https://github.com/jfbercher/labquiz.git#subdirectory=quiz_editor
```
with subdirectories `meta` (the bundle) , `quiz_nb` (for `labquiz`), `quiz_editor` and `quiz_dash`. 

---

# 🚀 Why LabQuiz?

LabQuiz is designed for **active learning** and **controlled assessment** in computational notebooks.

It helps instructors:

* Increase student engagement with embedded exercises
* Provide structured feedback during lab sessions
* Monitor progress in real time
* Run controlled tests and exams
* Detect configuration tampering or integrity violations

It helps students:

* Learn through interaction and immediate feedback
* Track their progress
* Work within structured assessment modes

---

# 🚠 What LabQuiz Does

Inside your notebook, you can:

* ✅ Add multiple-choice questions (`mcq`)
* 🔢 Add numerical questions with tolerance (`numeric`)
* 🧩 Create parameterized template questions
* 🔁 Limit attempts
* 💡 Provide hints and corrections
* 📊 Compute automatic scores
* 🌐 Log all activity to a Google Sheet backend (optional)
* 🔐 Enable exam mode with integrity checks

Example:

```python
from labquiz import QuizLab

quiz = QuizLab(URL, "my_quiz.yml", retries=2, exam_mode=False)
quiz.show("quiz1")
```
---

# 📸 Examples

## Multiple-choice question (with hints & correction)


![MCQ Example](doc_images/quiz2.gif)

## Numerical question

![Numeric Example](doc_images/quiz59.gif)

## Template-based question (dynamic variables)

![Template Example](doc_images/con_matrix_nb.gif)

![Template Example](doc_images/reglin_nb.gif)

![Template Example](doc_images/reglin_slope_nb.gif)

---

# 🧩 Question Types, Pedagogical modes, Logging

## Question types

LabQuiz supports four types:

| Type               | Description                           |
| ------------------ | ------------------------------------- |
| `mcq`              | Standard multiple-choice              |
| `numeric`          | Numerical answers with tolerance      |
| `mcq-template`     | Context-dependent MCQ                 |
| `numeric-template` | Context-dependent numerical questions |

**Template questions** allow dynamic evaluation based on runtime variables — ideal for practical lab computations.

Example:

```python
quiz.show("quiz54", a=res1, b=res2)
```

Variables can also be generated dynamically
```python
quiz.show("quiz54", autovars=True)
```

The expected solution is dynamically computed using Python expressions.

## Pedagogical modes

LabQuiz supports three pedagogical modes:

* Learning mode (hints + correction available, score display)
* Test mode (limited attempts, score display but no correction)
* Exam mode (no feedback, secure logging)

Quizzes are defined in simple YAML format and support

* Logical constraints (XOR, IMPLY, SAME, IMPLYFALSE)
* Bonuses and penalties
* Relative and absolute tolerances
* Variable generation for templates

## 📊 Remote Logging & Dashboard

All data can be stored in a **Google Sheet backend**. 

LabQuiz can log: Validation events, Parameters, User answers, Integrity hashes... 
LabQuiz also includes multiple anti-cheating mechanisms (Machine fingerprinting, Source hash verification, Detection of parameter tampering, Optional encrypted question files, Runtime integrity daemon...)


---

# ⚙️ Installation

## From PyPI

```bash
pip install labquiz
```

## From source

```bash
pip install git+https://github.com/jfbercher/labquiz.git
```

Import:

```python
import labquiz
from labquiz import QuizLab
```

Instantiate:

```python
quiz = QuizLab(URL, QUIZFILE,
               retries=2,
               needAuthentication=True,
               mandatoryInternet=False)
```

---


# 🛠 Additional Tools

## ✏️ `quiz_editor` — Build & Export Question Banks

Creating YAML files manually is possible — but **`quiz_editor` is intended to make it easier.** It can also be useful outside of LabQuiz as a general quiz-editor with export capabilities and large markdown support (including images, tables, equations, etc).

### Key features:

* Visual question editing (MCQ, numeric, templates) 
* Categories & tags
* Variable generation for templates
* Bonus / malus configuration
* Logical constraints (XOR, IMPLY, SAME, etc.)
* One-click export to:

  * ✅ YAML
  * 🔐 Encrypted version
  * 🌍 Interactive HTML (training mode), with dynamic support for regenerating variable values in template questions 
  * 📝 HTML exam version (Google Sheet connected)
  * 📄 Import-export to AMC–LaTeX format (paper exams), static and dynamic versions (using pythonTeX)
  * 📄 Import-export to Moodle XML format (LMS)

Online version:
👉 [https://jfb-quizeditor.streamlit.app/](https://jfb-quizeditor.streamlit.app/)

Install locally:

```bash
pip install quiz-editor
```

![Quiz Editor](doc_images/quiz_editor_2.png)


---

## 📊 `quiz_dash` — Real-Time Monitoring & Correction

`quiz_dash` is the companion dashboard for instructors.

It connects to your Google Sheet backend and provides:

* 📈 Live tracking of submissions
* Live class overview
* 👤 Student-by-student monitoring
* 🔍 Integrity checks (mode changes, retries tampering, hash verification)
* ⚖ Adjustable grading weights and scale
* 🔄 Automatic recalculation
* 📔 Full individual corrections available for download
* 📥 CSV export of results

Online version:
👉 [https://jfb-quizdash.streamlit.app/](https://jfb-quizdash.streamlit.app/)

![Dashboard](doc_images/Monitoring_quizzes_2.png)

![Dashboard](doc_images/Monitoring_marks.png)


---
## 🌍 Optional: Zero Installation with JupyterLite

LabQuiz can run entirely in the browser using JupyterLite (WASM).
Perfect for fully web-based lab environments.


# 📦 Ecosystem

| Tool            | Purpose                           |
| --------------- | --------------------------------- |
| **labquiz**     | Notebook quiz engine              |
| **quiz_editor** | Question bank creation & export   |
| **quiz_dash**   | Monitoring & correction dashboard |

📦 Repositories:

[https://github.com/jfbercher/labquiz](https://github.com/jfbercher/labquiz) is a multipackage repository, that includes the [bundle](https://github.com/jfbercher/labquiz/tree/main/meta), and

* [the `labquiz` notebook package](https://github.com/jfbercher/labquiz/tree/main/quiz_nb), 
* [the `quiz_editor` package](https://github.com/jfbercher/labquiz/tree/main/quiz_editor), 
* [the `quiz_editor` package](https://github.com/jfbercher/labquiz/tree/main/quiz_dash)

Online tools:

* [https://jfb-quizeditor.streamlit.app/](https://jfb-quizeditor.streamlit.app/)
* [https://jfb-quizdash.streamlit.app/](https://jfb-quizdash.streamlit.app/)

---


# 🎯 Typical Workflow

1. Prepare questions (YAML or `quiz_editor`)
2. Optionally encrypt file
3. Create Google Sheet backend
4. Instantiate `QuizLab` in notebook
5. Run lab / test / exam
6. Monitor using a python console or with `quiz_dash`
7. Post-correct with adjustable grading

---

# 🏁 Demonstration

See:

* `labQuizDemo_en.ipynb` in `extras/`
* 👉🏼 `Live version` 👈  Try it in [binder](https://mybinder.org/v2/gh/jfbercher/labquiz/main?urlpath=%2Fdoc%2Ftree%2Fextras%2FlabQuizDemo_en_binder.ipynb) 

---

# 📜 License

GPL-3.0 license

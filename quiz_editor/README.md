
[![PyPI version](https://img.shields.io/pypi/v/quiz-editor.svg)](https://pypi.org/project/quiz-editor/)
[![Python versions](https://img.shields.io/pypi/pyversions/quiz-editor.svg)](https://pypi.org/project/quiz-editor/)
[![License](https://img.shields.io/pypi/l/quiz-editor.svg)](https://pypi.org/project/quiz-editor/)

**Visual editor for creating, managing, and exporting LabQuiz question banks.**

`quiz_editor` is a Streamlit-based application designed to help instructors build structured quiz databases for use with **[LabQuiz](https://github.com/jfbercher/labquiz)** — without manually editing YAML files. `quiz_editor` can also be useful outside of LabQuiz as a general quiz-editor with export capabilities.

It supports question authoring, template variable simulation, encryption, tagging, and export to multiple formats.

---

## Why quiz_editor?

While LabQuiz uses YAML files for defining quizzes, editing them manually can become tedious for large question banks.

`quiz_editor` provides:

* A visual interface for question creation
* Support for multiple question types
* Built-in encryption & encoding
* Export to multiple formats (subset of questions, HTML, AMC-$\LaTeX$)
* Tagging, Categories and filtering system
* Template variable simulators

---

## Installation

### From PyPI

```bash
pip install quiz-editor
```

### From source

```bash
pip install git+https://github.com/jfbercher/quiz_editor.git
```

---

## Launch

An entry-point is provided that allows to simply use 
```bash
quiz_editor
```

Or use the hosted version:

👉 [https://jfb-quizeditor.streamlit.app/](https://jfb-quizeditor.streamlit.app/)

---

<div align="center">
  <img src="https://github.com/jfbercher/labquiz/raw/main/docs/doc_images/quiz_editor_2.png" width="80%" height="auto" alt="quiz_editor">

`quiz_editor` -- editing a question, with category, tags, choice of question type (multiple choice, numeric, etc.)
</div>  

<div align="center">
  <img src="https://github.com/jfbercher/labquiz/raw/main/docs/doc_images/quiz_editor_1.png" width="90%" height="auto" alt="quiz_editor">

`quiz_editor` -- editing a proposition -- correct or incorrect, hint (tip), answer (displayed during correction), bonus, penalty, etc.
</div>  


<div align="center">
  <img src="https://github.com/jfbercher/labquiz/raw/main/docs/doc_images/quiz_editor_3.png" width="90%" height="auto" alt="quiz_editor">

`quiz_editor` -- example of template usage with potentially external variables, including an editor preview based on automatically generated variables.
</div> 



<div align="center">
  <img src="https://github.com/jfbercher/labquiz/raw/main/docs/doc_images/exports1.png" width="90%" height="auto" alt="quiz_editor_exports">

`quiz_editor` -- Export examples (AMC-$\LaTeX$, interactive HTML, HTML-exam)
</div>  

<div align="center">
  <img src="https://github.com/jfbercher/labquiz/raw/main/docs/doc_images/exports2.png" width="40%" height="auto" alt="Centered image">
 
`quiz_editor` -- YAML Export with normal/encoded/encrypted options and question-only file
</div>  
---

## Supported Question Types

* `mcq` — Multiple choice
* `numeric` — Numerical with tolerance
* `mcq-template` — Parameterized MCQ
* `numeric-template` — Parameterized numeric

---

## Template Variable Simulation

For template questions, `quiz_editor` allows you to define **variable generators (“simulators”)** directly in the interface.

Each variable can specify:

* `type` (int, float, etc.)
* `structure` (scalar, vector, …)
* `engine` (e.g., NumPy RNG)
* `call` (generation function, e.g. `integers(0,10,size=1)`)

This enables:

* Dynamic generation of values
* Testing template behavior before export
* Automatic inclusion of variable definitions in the YAML file
* Consistent export to LabQuiz, HTML, or AMC formats

These simulators make template questions robust and reusable across sessions.

---

## Features

### Question Management

* Create / edit / delete questions
* Add categories and tags
* Define logical constraints (XOR, IMPLY, SAME, IMPLYFALSE)
* Configure bonuses and penalties
* Define tolerances for numerical answers
* Template variables simulation: define **variable generators** directly in the interface, which enable dynamic generation of values in LabQuiz and in HTML, or AMC exports.

---

### Export Options

`quiz_editor` can export to:

* ✅ YAML (LabQuiz format)
* 🔐 Encrypted version
* 🔎 Base64-encoded version
* 🌍 Interactive HTML (training mode)
* 📝 HTML exam version (Google Sheet connected)
* 📄 AMC–LaTeX format (for paper exams)

This allows reuse of the same question bank across:

* Notebook-based quizzes
* Web-based self-assessment
* Secure online exams
* Paper multiple-choice exams

---

## Integration with LabQuiz

Typical workflow:

1. Create questions in `quiz_editor`
2. Export encrypted YAML
3. Use file in:

```python
from labquiz import QuizLab
quiz = QuizLab(URL, "questions.yaml")
```

---

## Part of the LabQuiz ``Ecosystem''

* `labquiz` — Notebook quiz engine
* `quiz_editor` — Question bank editor
* `quiz_dash` — Monitoring & correction dashboard

---

# 📜 License

GNU GPL-3.0 license

---
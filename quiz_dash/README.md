
[![PyPI version](https://img.shields.io/pypi/v/quiz-dash.svg)](https://pypi.org/project/quiz-dash/)
[![Python versions](https://img.shields.io/pypi/pyversions/quiz-dash.svg)](https://pypi.org/project/quiz-dash/)
[![License](https://img.shields.io/pypi/l/quiz-dash.svg)](https://pypi.org/project/quiz-dash/)

**Real-time monitoring and correction dashboard for LabQuiz (and HTML exam export).**

`quiz_dash` is a Streamlit-based instructor dashboard that connects to a LabQuiz backend (Google Sheets) and provides live tracking, integrity verification, and automated correction.

It is designed for lab sessions, continuous assessment, and secure evaluation.

---

## Why quiz_dash?

When using LabQuiz in connected mode, all student interactions are logged.

Based on the Google Sheet URL specification, the associated read password, and the YAML file containing the answers, `quiz_dash`  allows instructors to:
 
- <mark>track over time</mark>, with an adjustable refresh rate, the submissions made by each participant, with the labels of the relevant quizzes,
- <mark>check integrity</mark>, i.e., verify that the parameters (number of attempts allowed, mode, etc.) have not been modified, verify the hash of the sources, the object in memory, and its dependencies, 
- <mark>view</mark>, over time, the progress of each participant (filterable) and of the entire group,
- <mark>correct</mark> and retrieve the results table, 
- with the possibility of adjusting the weight matrix (for multiple-choice questions) and the scoring scale per question,
- export structured results,
- export individual or whole class graduated scripts. 

---

## Installation

### From PyPI

```bash
pip install quiz-dash
```

### From source

```bash
pip install git+https://github.com/jfbercher/quiz_dash.git
```

---

## Launch

An entry-point is provided that allows to simply use 
```bash
quiz_dash
```
at the command line. 

Or use the hosted version:

👉 [https://jfb-quizdash.streamlit.app/](https://jfb-quizdash.streamlit.app/)

---

## Main Features

### 📈 Live Monitoring

* Track submissions over time
* View student-by-student activity
* Monitor class progress
* Adjustable refresh rate

---

### 🔐 Integrity Verification

Detect:

* Mode changes (exam/test/learning)
* Modified retry limits
* Source hash changes
* Machine sharing
* Kernel restarts

Designed to support controlled exam conditions.

---

### ⚖ Flexible Correction

* Automatic correction
* Adjustable weight matrix
* Per-question grading scale
* Recalculation on demand
* Threshold control (negative scores allowed or not)

---

### 📥 Export

* Download corrected results as CSV
* Recompute scores with updated parameters
* Export individual PDF or HTML reports

---

Some screenshots of actual monitoring:
<div align="center">
  <img src="https://github.com/jfbercher/labquiz/raw/main/docs/doc_images/dash_parameters.png" width="80%" height="auto" alt="dash_parameters">
 
`quiz_dash` -- Dash parameters
</div>  

<div align="center">
  <img src="https://github.com/jfbercher/labquiz/raw/main/docs/doc_images/Monitoring_integrity.png" width="80%" height="auto" alt="Monitoring_integrity">

`quiz_dash` -- Integrity monitoring
</div>  

<div align="center">
  <img src="https://github.com/jfbercher/labquiz/raw/main/docs/doc_images/Monitoring_quizzes_2.png" width="50%" height="auto" alt="Monitoring_activity">

`quiz_dash` -- Monitoring of quizzes taken by students and the group. Automatic refresh possible and adjustable (student names have been hidden)
</div>  

<div align="center">
  <img src="https://github.com/jfbercher/labquiz/raw/main/docs/doc_images/Monitoring_marks.png" width="80%" height="auto" alt="Monitoring_marks image">

`quiz_dash` -- Automated correction, with the option to adjust the scoring system (automatic recalculation); (student names have been hidden). Of course, the results table can be downloaded.  
</div>  

<div align="center">
  <img src="https://github.com/jfbercher/labquiz/raw/main/docs/doc_images/Individual_correction.png" width="80%" height="auto" alt="Monitoring_marks image">

`quiz_dash` -- `quiz_dash` -- Individual correction can be generated for a student.
</div>  

<div align="center">
  <img src="https://github.com/jfbercher/labquiz/raw/main/docs/doc_images/Zipping_all_pdf_reports.png" width="80%" height="auto" alt="Monitoring_marks image">

`quiz_dash` --  Individual reports can be downloaded so that you can distribute them to the students. 
</div> 


## Typical Workflow

1. Students run quizzes in Jupyter notebooks (LabQuiz)
2. Results are logged to Google Sheets
3. Instructor opens `quiz_dash`
4. Monitor activity in real time
5. Run correction
6. Export final grades

---

## Integration

`quiz_dash` works with:

* `labquiz` — quiz engine
* `quiz_editor` — question preparation tool

---

## Intended Use Cases

* Lab session monitoring
* Continuous assessment
* Exam supervision
* Integrity auditing
* Post-session grade recalculation

---

# 📜 License

GNU GPL-3.0 license

---
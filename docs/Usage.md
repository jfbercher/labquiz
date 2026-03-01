---
title: Installation & Usage
subtitle: 'LabQuiz: A suite of tools for integrating quizzes into Jupyter notebooks'
date: 02/27/2026
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

# 🛠 Installation  
From source:

```bash
pip install git+https://github.com/jfbercher/labquiz.git
```

## From PyPI

```bash
pip install labquiz
```

in a terminal, or `%pip ...` in a notebook cell.
It is possible to use a standalone version, *without any Python installation*, which runs in the browser. This is described in [](jupyterlite).

(usage)=
# Usage
Once labquiz is installed, import it with
```{code}
import labquiz
from labquiz import QuizLab
```
and instantiate a quiz with
```{code}
URL = "" # path to a URL to collect results
QUIZFILE = "name_of_quiz_file" # for example "quizzes_basic_filtering_test.yaml"
quiz = QuizLab(URL, QUIZFILE)
```
Additional parameters can be specified (default values below)
```
needAuthentication=True,  # Authentication required yes/no
retries=2,                  # number of possible attempts = retries + 1
mandatoryInternet=False,    # requirement to have a valid internet connection
CHECKALIVE=60,              # integrity check every CHECKALIVE seconds
#(no changes to the program or parameters)
```
                            
:::{figure} doc_images/login.gif
:name: login
:label:fig
:alt: Login
:align: left
:width: 60%
Login example
:::
# Features
⚠️ LabQuiz also works in **visual studio code**, but $\LaTeX$ portions are not rendered in questions, suggestions, and answers. This is a limitation of Visual Studio, which does not load MathJax, and may be improved in the future. In the meantime, it is better to use a classic Jupyter, Colab, or JupyterLite. In Visual Studio, this can be managed, but it is less effective if there is $\LaTeX$ in the text. 
## Types of quiz questions
Four types of questions are available:
 
- multiple-choice questions (`type: "qcm"` in the question file; this is the default type)
- numerical questions (`type: "numeric"`)
- Context-dependent multiple-choice questions (`type: "qcm-template"`)
- Context-dependent numerical questions (`type: "numeric-template"`)
The structure of the question file, where these different types are indicated, is presented [](question_file_structure). How to prepare and even encrypt this file is presented in [](prepare_encode_file).
 
Templates allow you to ask questions that depend on local variables. This allows you to test specific values, orders of magnitude, the result of a calculation, or the consistency between several values. Here are two examples to illustrate this:
:::{figure} doc_images/quiz5354.gif
:name: quiz3
:label:fig3
:alt: Quiz example
:align: center
:width: 60%
Template questions that use numerical values passed as parameters
:::

## Different presentation modes
### Learning mode
In learning mode, all four buttons are available. The validate button displays the score obtained, see [](fig5). The tips button displays advice. The correct button displays the correction, see [](fig6) and [](fig7). The boxes turn green if they have been checked or unchecked correctly, and red if not. The checks entered by the user are retained, but colored green or red depending on the correct result. After the correction has been requested, the validate, reset, and tips buttons are disabled and become inoperative.
 
:::{figure} doc_images/4buttons_submit_actif.png
:name: quiz3
:label:fig5
:alt: Quiz example
:align: left
:width: 60%
Learning mode - the submit button has been pressed
:::
:::{figure} doc_images/4buttons_correction_active.png
:name: quiz3
:label:fig6
:alt: Quiz example
:align: left
:width: 60%
Learning mode - the corrected button has been pressed
:::

:::{figure} doc_images/exemple_correction.png
:name: quiz3
:label:fig7
:alt: Quiz example
:align: left
:width: 60%
Learning mode - the correct button has been pressed and the correction is displayed
:::
### Test mode
In test mode, the correct button is removed. The student sees their score after validation and tips are possible. The number of validations is limited by the `retries` parameter passed at initialization.
:::{figure} doc_images/3buttons_submit_active.png
:name: quiz3
:label:fig8
:alt: Quiz example
:align: left
:width: 60%
Test mode - the validate button has been pressed
:::
### Exam mode
In exam mode, there is no score display, no tips, and no corrections. Only the reset and validate buttons appear.
:::{figure} doc_images/2buttons_submit_active.png
:name: quiz3
:label:fig9
:alt: Quiz example
:align: left
:width: 40%
Exam mode - the submit button has been pressed
:::
### Individual questions or series of questions `exam_show`
Regardless of the presentation modes described above, questions can be presented either individually with a call of the form `quiz.show(‘label’)`, or in a block of questions. The latter option can be useful for a short interim review, for example.
 
A block of questions will be presented using the `quiz.exam_show()` function with the following parameters:
```
exam_show(exam_title="", questions=None, shuffle=False, nb=None)
```
- exam_title: Used to identify the exam, default ""
- questions:  List of question labels (e.g. "quiz1," "quiz2").
             If None, uses all questions in the bank.
- shuffle: default False; If True, shuffles the order of questions before display.
- nb: If not None, randomly draws nb distinct questions
            from the set of available questions.
            
An example of a call could be 
```
ql = quiz.exam_show(exam_title="Test to see", shuffle=True, nb=4)
```
Intermediate results are not displayed (if in learning or test mode). 
The results obtained can then be viewed by the student with `quiz.exam_result(ql, bareme=None)`.
 
In exam mode, results are not calculated or viewable, and the teacher can correct all exams as described in [](correction_par_enseignant).

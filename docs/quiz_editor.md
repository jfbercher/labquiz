---
title: quiz_editor
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

# `quiz_editor` - Prepare and encode the question file - 🧑‍🏫 🏫 

Manual preparation is possible for small databases, but a specific editor has also been created to allow editing and various operations (extraction, encoding, encryption, conversion to other formats).
 
This editor is available here:
https://jfb-quizeditor.streamlit.app/

🛠  It can also be installed from pypi by
```bash
pip install quiz-editor 
``` 
🛠 or from git 
```bash
pip install git+https://github.com/jfbercher/quiz_editor.git
``` 


It allows you to edit the files whose structure has been described and has some additional features:
- ability to define {underline}`categories` and {underline}`tags`, and to select questions by filtering on categories and tags,
- define variables generation for templates,
- extraction and saving of some of the questions,
- preparation of an encoded or encrypted version (as described above),
- <mark>conversion to AMC-$\LaTeX$ </mark> format, so that questions can be reused in a paper-based AMC multiple-choice quiz,
 
(Categories are used to define the `\element` type), an example of output [here](https://www.esiee.fr/~bercherj/labquizDemo/files/demo_export/demo_export.tex)
- <mark>conversion to HTML format</mark>
  with integrated answers, so that a "self-assessment" web page can be created -- an example of the export result [here](https://www.esiee.fr/~bercherj/labquizDemo/files/demo_export/demo_export_train.html)
- conversion to HTML format **exam**, without answers, and with real-time submission of results to a Google Sheet, with subsequent correction, as described in [](calcul_du_score_correction) or using the dashboard as described below [](quiz_dash)  -- example [here](https://www.esiee.fr/~bercherj/labquizDemo/files/demo_export/demo_export_exam.html)
  
:::{figure} doc_images/quiz_editor_2.png
:name: quiz_editor_1
:label:fig24
:alt: quiz_editor
:align: center
:width: 90%
`quiz_editor` -- editing a question, with category, tags, choice of question type (multiple choice, numeric, etc.)
:::
:::{figure} doc_images/quiz_editor_1.png
:name: quiz_editor_2
:label:fig25
:alt: quiz_editor
:align: center
:width: 90%
`quiz_editor` -- editing a proposition -- correct or incorrect, hint (tip), answer (displayed during correction), bonus, penalty, etc.
:::
:::{figure} doc_images/exports1.png
:name: quiz_editor_exports
:label:fig26
:alt: quiz_editor_exports
:align: center

`quiz_editor` -- Export examples (AMC-$\LaTeX$, interactive HTML, HTML-exam)
:::

:::{figure} doc_images/exports2.png
:name: quiz_editor_exports_bis
:label:fig27
:alt: quiz_editor_exports
:align: center
:width: 40%

`quiz_editor` -- YAML Export with normal/encoded/encrypted options and question-only file
:::

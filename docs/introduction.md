---
title: Introduction & first examples
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

```{note} LabQuiz in a nutshell
 
 It is a Python package that
 
- allows you to integrate questions in the form of multiple-choice questions or numerical values,
 - with multiple attempts (adjustable), hints, and *feedback*,
 - correct quizzes online or *a posteriori*, 
 - automatically authenticates and records responses on a remote server,
 - provides a dashboard to track student progress,
 - and more
```

```{tip} Learning objectives
- encourage student engagement through exercises and *fun* assessment,
- monitor and measure the results to be achieved during the session,
- allow students to *take a step back* by linking the results to the course and going into more depth,
- encourage learning through progress indicators and *feedback*,
- monitor, using using a *dashboard*, monitor student progress over time,
- as well as the entire group,
````

```{seealso} Links
**Source code**

- https://github.com/jfbercher/labquiz
- https://github.com/jfbercher/quiz_editor
- https://github.com/jfbercher/quiz_dash

**Online**

- https://jfb-quizeditor.streamlit.app/
- https://jfb-quizdash.streamlit.app/
```

```{hint} Table of contents
To display a table of contents, press the table of contents button on the left; or menu `View/Table of Contents`
```

# First examples

*Quizzes* are integrated into the subject throughout the tutorial or practical exercise, based on a questionnaire prepared in advance.
A first example to illustrate a multiple-choice quiz, with tips and corrections:
 
:::{figure} doc_images/quiz2.gif
:label: fig1
:name: quiz2
:alt: Quiz example
:align: center
:width: 60%
Question during the lab session
:::

Second example with numerical values
:::{figure} doc_images/quiz59.gif
:name: quiz2
:label:fig2
:alt: Quiz example
:align: center
:width: 60%
Another question during the practical work
:::

(with the quiz blocked if the answer has already been submitted! 😊)
```{warning} Connected mode
**In connected mode**, i.e. as soon as a valid URL has been passed during initialization
- all entries (Validate, Correct) are recorded and transmitted,
- a specific identifier for each machine is calculated and used to identify all logs,
- a system status is transmitted periodically, with a known and verifiable period, allowing the detection of
- changes to settings (exam mode, number of attempts allowed, etc.),
- changes to the package source,
    
- code injection, 
    - a kernel restart, a new instantiation, a machine change...
- etc
Of course, any interruption in periodic transmission by a duly registered machine is necessarily detected...
````

In the following
- the label 🧑‍🏫 🏫 refers to information specific to teachers or subject designers
- the label 🧑‍🎓 refers to information specific to students

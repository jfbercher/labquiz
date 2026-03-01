---
title: quiz_dash
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

(quiz_dash)=
#  `quiz_dash` - Monitor in real time and correct with `quiz_dash` - 🧑‍🏫🏫
As described above, you can load the results table at the terminal level and perform all the tests and statistics you want. However, it is easier to use a small graphical utility to do this. This monitoring and analysis dashboard is available here:
https://jfb-quizdash.streamlit.app/
Based on the Google Sheet URL specification, the associated read password, and the YAML file containing the answers, this dashboard allows you to:
 
- <mark>track over time</mark>, with an adjustable refresh rate, the submissions made by each participant, with the labels of the relevant quizzes,
- <mark>check integrity</mark>, i.e., verify that the parameters (number of attempts allowed, mode, etc.) have not been modified, verify the hash of the sources, the object in memory, and its dependencies, 
- <mark>view</mark>, over time, the progress of each participant (filterable) and of the entire group,
- <mark>correct</mark> and retrieve the results table, 
- with the possibility of adjusting the weight matrix (for multiple-choice questions) and the scoring scale per question. .
Some screenshots of actual monitoring:
:::{figure} doc_images/dash_parameters.png
:name: dash_parameters
:label:fig26
:alt: quiz_dash
:align: center
:width: 90%
`quiz_dash` -- Entering monitoring/correction configuration parameters
:::
:::{figure} doc_images/Monitoring_integrity.png
:name: Monitoring_integrity
:label:fig27
:alt: quiz_dash
:align: center
:width: 90%
`quiz_dash` -- Integrity monitoring
:::
:::{figure} doc_images/Monitoring_quizzes.png
:name: Monitoring_activity
:label:fig28
:alt: quiz_dash
:align: center
:width: 90%
`quiz_dash` -- Monitoring of quizzes taken by students and the group. Automatic refresh possible and adjustable (student names have been hidden)
:::
:::{figure} doc_images/Monitoring_marks.png
:name: Monitoring_marks
:label:fig29
:alt: quiz_dash
:align: center
:width: 90%
`quiz_dash` -- Automated correction, with the option to adjust the scoring system (automatic recalculation); (student names have been hidden). Of course, the results table can be downloaded.  
:::

---
title: 'LabQuiz: A Quiz-as-Code Ecosystem for Integrated Assessment and Student Engagement in Jupyter Notebooks'
tags:
  - Python
  - Jupyter
  - Education
  - Formative Assessment
  - YAML
  - Pedagogy
authors:
  - name: Jean-François Bercher
    email: jf.bercher@esiee.fr
    url: https://perso.esiee.fr/~bercherj/
    affiliation: "1, 2" 
    orcid: 0009-0007-5474-7475
affiliations:
 - name: ESIEE Paris - Université Gustave Eiffel
   index: 1
 - name: Laboratoire d'informatique Gaspard Monge
   url: https://ligm.univ-eiffel.fr/
   index: 2
date: 24 March 2026
bibliography: paper.bib
---


## Summary

`LabQuiz` is an open-source ecosystem designed to integrate structured assessment directly into Jupyter-based computational learning environments to foster student engagement. It combines a Python package for embedding interactive quizzes into notebooks, a graphical interface for authoring parameterized question banks (`quiz_editor`), and a monitoring and grading dashboard for tracking student activity (`quiz_dash`).

The system is built around a *Quiz-as-Code* approach, where quizzes are defined in a structured YAML format and rendering them as interactive components using `ipywidgets`. This design allows quizzes to be versioned, reused, and deployed across multiple contexts while remaining integrated with computational workflows. By unifying content authoring, delivery, and analysis, `LabQuiz` supports both *formative, continuous assessment* and *summative evaluation*, preserving alignment with computational practices in modern STEM education.

## Statement of Need

Formative assessment and low-stakes testing are widely recognized as key drivers of engagement and long-term retention [@Roediger2006]. In modern STEM education, Jupyter notebooks have become a central medium for active learning, particularly in data science, engineering, and statistics. Their strength lies in integrating narrative text, executable code, and visualization. However, despite the notebooks' interactive nature, the assessment phase is often delegated to external platforms such as Learning Management Systems (LMS).

This disconnect introduces a significant *pedagogical discontinuity*: students construct knowledge in an interactive setting but are evaluated in a separate interface that cannot access their work state. In computational disciplines, this separation weakens the link between learning and assessment. External systems cannot evaluate results that depend on dynamic computations or intermediate steps performed within the notebook, leading to simplified question formats that do not reflect the complexity of computational tasks.

Implementing frequent formative assessment at scale presents logistical challenges for educators. Effective solutions require tools that are easy to author and adaptable across various teaching contexts, including laboratory sessions, interactive lessons, and self-paced practice. By decoupling quiz definition from rendering and deployment, `LabQuiz` allows educators to embed sophisticated, interactive assessments directly within the computational workflow. This approach ensures that assessment becomes a continuous, engaging part of the learning journey.

![Overview of the `LabQuiz` ecosystem. Dashed arrows indicate optional components. YAML quizzes drive both interactive notebooks and exports, monitored via a dashboard producing analytics, mark tables, and student reports.](mermaid-fig3.png){ width=100% }


## Availability and Licensing

The `LabQuiz` ecosystem is released under GNU GPL-3.0 license. The [project](https://github.com/jfbercher/labquiz) is modular, consisting of the core Python package, the graphical `quiz_editor`, and the `quiz_dash` monitoring dashboard [@Bercher_LabQuiz_2026].

Comprehensive documentation is available on [ReadTheDocs](https://labquiz.readthedocs.io/en/latest/). To lower the barrier to entry, the project provides "batteries-included" examples and demonstration environments through [Binder](https://mybinder.org/v2/gh/jfbercher/labquiz/main?urlpath=%2Fdoc%2Ftree%2Fextras%2FlabQuizDemo_en_binder.ipynb) and [JupyterLite](https://perso.esiee.fr/~bercherj/labquizDemo/lab/index.html?path=labQuizDemo_en.ipynb), enabling educators to explore the full system directly in a web browser without installation.

## Relationship to Existing Tools

Several tools address assessment in Jupyter, but they typically target specific niches. Tools such as `nbgrader` [@Hamrick2019] support autograding of programming assignments but focus on batch submission workflows rather than interactive in-session quizzes or real-time monitoring.

Another related tool, `jupyter-quizzes` [@Shea_JupyterQuiz_2025], supports lightweight embedding of multiple-choice and numerical questions. However, as it is primarily intended for interactive textbooks, it does not provide runtime parameterization, backend logging, or dedicated monitoring dashboards for instructors. `LabQuiz` complements these systems by emphasizing continuity and flexibility across diverse teaching contexts, particularly for live classroom management.

## Design and Implementation

At the core of `LabQuiz` is the principle that assessment should be a first-class component of the computational environment. Quizzes are defined using a structured YAML format, as a versionable source of truth.

These definitions are rendered within Jupyter notebooks [@Kluyver2016] using `ipywidgets`. This allows questions to depend on computed values, ensuring alignment with the underlying code. Additionally, since the system supports *parameterized questions*, instructors can define templates with dynamically generated parameters, enabling the creation of varied instances.

The ecosystem supports the full assessment workflow, combining a graphical editor (`quiz_editor`), a lightweight remote backend (e.g., a shared spreadsheet) for storing responses, and a monitoring dashboard (`quiz_dash`) that provides real-time access to student progress.

The architecture is compatible with standard Jupyter environments and JupyterLite, enabling *zero-install deployment* on a wide range of devices, including tablets and smartphones.

## Educational Value

The core value of `LabQuiz` lies in strengthening the feedback loop between students and instructors [@Hattie2007]. By embedding quizzes within notebooks, the system restores *pedagogical continuity*, reducing cognitive load by keeping a single workspace [@Sweller1994].

The system extends *interactivity* by allowing questions to reflect computational tasks. This enables continuous formative assessment, where immediate feedback supports iterative engagement in digital learning [@kluger1996effects], [@bjork_self-regulated_2013], [@ramadhan2025exploring] and consistent with "test-enhanced learning" [@Roediger2006].

For instructors, real-time data facilitates *responsive teaching*. Rather than relying solely on post-hoc evaluation, instructors can monitor student progression and adjust guidance accordingly [@Gasevic2016]. This shifts the focus from assessment as measurement to assessment as an integral component of the learning process [@Chassignol2018].

## Classroom Implementation and Experience

`LabQuiz` was first introduced in computational lab sessions to promote active exploration. Embedding quizzes regularly prompts students to test their understanding, reducing passive progression through the material.

The system has been successfully deployed with cohorts of 100+ students across four parallel laboratory tracks on several occasions. It enabled real-time monitoring of student progression, identification of common bottlenecks, and "just-in-time" group clarifications, as well as post-hoc assessment through analytics. Instructors were able to reconstruct individual performance, generate graded reports for students, and ensure consistency across parallel sessions.

The separation between definition, rendering, and configuration allows the same quiz to be deployed across contexts, from guided sessions to self-contained interactive lessons. Instructors can move fluidly between formative exploration and summative evaluation within the same session by adjusting configuration (e.g., feedback visibility, attempt limits). This flexibility reinforces the coherence of the overall learning experience.

## Conclusion and Future Directions

`LabQuiz` provides a robust framework for Jupyter-based assessment. By adopting a "Quiz-as-Code" workflow, it enables educators to maintain high-quality materials at scale. Its success with large cohorts demonstrates that immediate feedback and monitoring enhance student engagement.

Future development will focus on extending export capabilities (e.g., LMS formats), enhancing analytics, and supporting community contributions.

## References

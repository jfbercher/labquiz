---
title: Deploy to Jupyterlite
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


[Jupyterlite](https://jupyterlite.readthedocs.io/en/stable/index.html) is a Python/ Jupyterlab distribution that runs entirely in the browser. The various Python packages have been ported to WebAssembly (WASM), a binary format that is executable in browsers. For optimized libraries (numpy, pandas, etc.), performance is equivalent to native performance. For pure Python (loops, etc.), execution is 3 to 10 times slower. What this means, however, is that you can prepare Python programs and Jupyter notebooks that are fully executable in a browser, with zero installation! And it works independently of the system, even on tablets, phones, etc.
For example, you can find here: [https://perso.esiee.fr/~bercherj/JliteNotes/lab/](https://perso.esiee.fr/~bercherj/JliteNotes/lab/) an example of a practical assignment distributed in this way, incorporating labquiz. 

(jupyterlite)=
# Deploy to Jupyterlite - 🧑‍🏫🏫

Preparation and deployment under Jupyterlite are not exactly immediate (there are documents to follow and adaptations to make), so I suggest a shortcut, which is as follows:
1. Download the archive of a deployed version, available here <br>
https://drive.google.com/file/d/14BnzVmPO6I8uOMEmNC6BTMsjXL3RLrQL/view?usp=sharing
<br> and unzip it on your disk,
 1. Place the files you want to make available in `_output/files`
1. Run the program `update_jupyterlite_contents.py` available in the `extras` directory by typing `python update_jupyterlite_contents.py`. Install dependencies if necessary. 
> 👉🏼 This will update the lists and characteristics of the files in the distribution
1. **init**) Upload *all* of the contents of `_output` to your web account, typically (or for instance...) to `login/public_html`, in a directory such as `MyJlite`, and you will obtain a distribution that is accessible and executable at `https://www.esiee.fr/~login/MyJlite`. If you don't know how to do this, ask your nearest local IT guru. 
2. **update**) If you only need to perform an update, simply copy the contents of `_output/files` **and** `_output/api` to `login/public_html/MyJlite/files` and `login/public_html/MyJlite/api`. 
For subsequent updates, add or modify files, you only need to perform steps 2, 3, and 5.maj); this will save you from having to upload the 70 MB in step 4.init). 


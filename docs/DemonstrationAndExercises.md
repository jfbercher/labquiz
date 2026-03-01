---
title: Demonstration and exercises
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

# Demonstration and exercises
Explore the demo notebook `labQuizDemo.ipynb` available in the `extras` directory and experiment. You can also use the live [binder version](https://mybinder.org/v2/gh/jfbercher/labquiz/main?urlpath=%2Fdoc%2Ftree%2Fextras%2FlabQuizDemo_en_binder.ipynb) or even the jupyterlite version available [here](https://perso.esiee.fr/~bercherj/labquizDemo/lab/index.html?path=labQuizDemo.ipynb)
 
If you want to go further, and if you've made it this far, you're probably motivated, so here are a few ideas.
  

- **Exercise**: Create a Google Sheet named `MyFirstGSQuiz` by following the instructions in [](google-sheet).  Change the read password in the first line of the script accessible via Extensions/Apps Script, then deploy. Note the deployment address and password.
- **Exercise**: Load or create a question database in `quiz_editor`. Add a few questions. Save the new file. Export an HTML version and an AMC version. Upload the HTML to your account and test (or test locally).
- **Exercise**: Export an encrypted version of your questions (note the password and update it in the Google Sheet, Config page) .
- **Exercise**: Create a Jupyter notebook, import labquiz, instantiate a quiz, -- see [](usage), integrate a few questions and test.
- **Exercise**: Instantiate a quiz linked to the URL of your Google Sheet, integrate a few questions into your notebook, and check that the answers appear in the Google Sheet.
- **Exercise**: Use `quiz_dash` to read the results. Change the settings in your notebook (change `mode_examen` from True to False, modify `retries`), and see if this triggers an alert. Run the correction on your few tests. 
- **Exercise**: Create a jupyterlite version as explained in [](jupyterlite), add your files, upload it to your info account in `public_html`, and check by connecting to the corresponding web address.
---
title: Manual preparation of question file
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

# Prepare and encode the question file (Manual preparation) - 🧑‍🏫 🏫 

The question file is a text file, so it can be prepared and maintained using a simple text editor, following the structure detailed in [](question_file_structure). Once this file has been prepared, you may want to generate
- [enc] a base64-encoded version (to prevent it from being too easily viewed)
- [crypt] an encrypted version with a hidden key
- [qo] a "questions only" version without tips and answers
In particular, 
- For an exam, you will distribute the "questions only" file, encrypted or not (you can also use it outside of an exam, but in this case the students will have no tips, answers, or scores)
- For a test mode where you plan to provide *feedback* to students, use the version with answers, in encoded or encrypted format, adjust `retries` and set `test_mode=True` in the parameters passed to the initialization.
A function allows you to perform these different operations:
```python
prepare_files(input_file, output_file, mode="crypt", pwd=‘’):    
    """
    
Prepare YAML files for quizzes. 
    Outputs two files, with the basename given in `output_file`.
 
    The second file is questions only and is the input stripped
    from responses and tips. With the `mode="crypt"`, the input and stripped
     
versions are encrypted; with `mode="enc"`, both files are base64 encoded; 
    finally, with `mode=yml`, files are not encoded nor encrypted.
Parameters
    
----------
input_file : str
Path to the input YAML file.
output_file : str
Path to the output YAML file.
mode : str, optional
Mode to prepare the file. Choose from "crypt", ‘enc’, or "yml".
pwd : str, optional
Password for file encryption.
```
It can be used, for example, as follows
```python
from labquiz.putils import prepare_files
prepare_files("quizzes_basic_filtering_test.yaml", "qbf.yml", mode="crypt", pwd=PASSWORD)
```
which produces
```
- Creating qbf_crypt.txt
- Creating qbf_qo_crypt.txt
⚠️ File encrypted with pwd. Ensure to use the `mandatoryInternet=True` option in quiz init
```
➡️  You will copy the password, if any, into cell A2 of the Config sheet of the Google Sheet, see [](google-sheet), if you are using a GS to collect the results.
 

- For an exam, you will distribute the "questions only" file, encrypted or not (you can also use it outside of an exam, but in this case, students will not have tips, answers, or scores).
- For a test mode where you plan to provide * feedback* to students, use the version with answers, in encoded or encrypted format, adjust `retries` and set `test_mode=True` in the parameters passed to the initialization.
---
title: Security
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
# Security

## Principles
Various measures have been taken to ensure a reasonable level of contributor identification, compliance with instructions, and fraud prevention. These measures include 
- Identification of the machine (via its software and hardware system), which provides an identifier used in all transactions.
- The quiz settings and status are retained if the quiz is re-instantiated. Only a kernel restart can reset everything (and in this case, all local data is also lost). .
- Encryption of the question file: this is encrypted using a key calculated at runtime and possibly dependent on a key available on a remote server, parameter `mandatoryInternet=True`.
 
- In connected mode, which is mandatory if the `mandatoryInternet` parameter has been set, *all* validations and correction requests are recorded and transmitted, along with the answers given to the current quiz, which also allows for post-correction.
- A hash of the sources and parameters is transmitted, allowing for the detection of modifications to the sources -- including monkey patching, or monitored parameters.
 
- A *daemon* also periodically transmits the system status, including, of course, the machine's identification.
🧑‍🏫🏫 - ⚠️ - That being said, to conduct an *exam* under good conditions, use a question file WITHOUT the answers. As soon as there are answers, even if they are difficult to access, a motivated student will always manage to obtain them (and share them). If you want to evaluate during a practical, either distribute only the file without answers, or use a very small coefficient.
 

## Detecting anomalies - 🧑‍🏫 🏫 
Functions have been prepared for this purpose. They allow you to test integrity at startup (`start`) or over time, by isolating the changes made, or by reporting that the hash is not as expected or has been modified. Here are some examples of implementation 
### Integrity
```python
from labquiz import QuizLab
from labquiz.putils import readData, getAllStudentsAnsvers, correctAll
# # 1 - Reading data
URL = "https://URL_UTILISÉE_POUR_RECUEILLIR_LES_RÉSULTATS"
SECRET = "SECRET_PASSWORD_SPECIFIED_IN_THE_SHEET"
data, data_filt = readData(URL, SECRET)
## 2 - Students
students = sorted(list(data["student"].dropna().unique()))
## 3 - imports 
from labquiz.putils import  check_integrity_all_std, check_start_integrity_all_std, check_hash_integrity
## 4 - test
originalp = {‘retries’:2, ‘exam_mode’:True, ‘test_mode’:False}
starting_values = {‘exam_mode’:True, ‘retries’:2 }
print("------------------------")
print("==> test start_integrity")
print("------------------------")
check_start_integrity_all_std(starting_values, data)
print("----------------------- ---------")
print("==> test check_integrity_all_std")
print("--------------------------------")
check_integrity_all_std(originalp, students, data)
```
The outputs are as follows (truncated with the indication [...])
```
------------------------
==> test check_integrity
------------------------
Dufour Lea: Machine modification for the same student name
[‘21b1f154204eb9e7’ '04b1f154204fa9e9']
Dufour Lea - record 13:
      - Original key ‘exam_mode’ changed from True to False
      
- Original key ‘test_mode’ changed from False to True
Dufour Lea - record 14:
- Original key ‘exam_mode’ changed from True to False
- Original key ‘test_mode’ changed from False to True
Dufour Lea - record 15:
- Original key ‘exam_mode’ changed from True to False
      
- Original key ‘test_mode’ changed from False to True
Dufour Léa - record 16:
- Original key ‘exam_mode’ changed from True to False
- Original key ‘test_mode’ changed from False to True
[...]
-------- ----------------
==> test start_integrity
------------------------
04b1f154204fa9e9 - record 42: Original key ‘exam_mode’ changed from True to False
04b1f154204fa9e9 - record 43: Original key ‘exam_mode’ changed from True to False
[...]
04b1f154204fa9e9 - record 53: Original ‘exam_mode’ key changed from True to False
21b1f154204eb9e7 - record 13: Original ‘exam_mode’ key changed from True to False
04b1f154204fa9e9 - record 27: Original ‘exam_mode’ key changed from True to False
----------------- ---------------
==> test check_integrity_all_std
--------------------------------
Dufour Léa: Machine modification for the same student name
[‘21b1f154204eb9e7’ '04b1f154204fa9e9']
Dufour Léa - record 13:
      - Original '
 
exam_mode' key changed from True to False
      - Original ‘test_mode’ key changed from False to True
Dufour Léa - record 14:
      - Original ‘exam_mode’ key changed from True to False
      - Original ‘test_mode’ key changed from False to True
Dufour Léa - record 15:
      - Original 'exam_ mode' changed from True to False
      - Original key ‘test_mode’ changed from False to True
[...]
J Jean-Marc - record 54:
- Original key ‘exam_mode’ changed from True to False
- Original key ‘test_mode’ changed from False to True
[...]
L Bobby - record 48:
      
- Original key ‘exam_mode’ changed from True to False
      - Original key ‘test_mode’ changed from False to True
L Bobby - record 49:
- Original key ‘exam_mode’ changed from True to False
- Original key ‘test_mode’ changed from False to True
L Bobby - record 50:
- Original key ‘exam_mode’ changed from True to False
- Original key ‘test_mode’ changed from False to True
L Bobby - record 51:
- Original key ‘exam_mode’ changed from True to False
- Original key ‘test_mode’ changed from False to True
Lamarq Linda - record 30:
      - Original ‘retries’ key changed from 2 to 2000000
Lamarq Linda - record 31:
- Original ‘retries’ key changed from 2 to 2000000
Lamarq Linda - record 32:
- Original ‘retries’ key changed from 2 to 2000000
Lamarq Linda - record 33:
- Original ‘retries’ key changed from 2 to 2000000
```
### Machine sharing
```python
"""
Demo: Detects whether the same machine has been used for multiple student names
"""
from labquiz.putils import check_machine
check_machine(data_filt) #Detects whether the same machine has been used for multiple student names
```
```
Same machine 04b1f154204fa9e9 used by several students
[‘Morane Bob’ 'L Bobby' ‘J Jean-Marc’ 'Legrand John' ‘Dufour Lea’
 ‘Makhoul Alain’ 'Lamarq Linda']
````
### Hash check
Identification of the expected hash -- the `retries`, `exam_mode`, and `test_mode` parameters must be the same as those imposed on students. 
```python
from labquiz.utils import get_full_object_hash
QUIZFILE = "NAME_OF_ORIGINAL_FILE" 
URL = ""
quiz = QuizLab(URL,
 
QUIZFILE, needAuthentication=False, mandatoryInternet=False, retries=2, exam_mode=True)
wanted_hash = get_full_object_hash(quiz,  modules = [‘main’, ‘utils’],
        WATCHLIST=[‘exam_mode’, ‘test_mode’, ‘retries’])
```
And test for everyone
```python
from labquiz.putils import check_hash_integrity
check_hash_integrity(data, ‘full’, wanted_hash=wanted_hash)
```
Which provides
```
⚠️ Dufour Léa, machine id 04b1f154204fa9e9:
    
👉🏼 The source or parameters have been modified or monkey patched
⚠️ index 17 hash: ddd8b0e5a3b35c41bb8db16cda874ff52af40a9ffdf56d7510abed65a9dec69f
⚠️ Dufour Léa, machine id 21b1 f154204eb9e7:
    👉 🏼 The source or parameters have been modified or monkey patched
hash found:
fb85b7bceb28ecce4a047d1fb94428c60789505ebd89e22a5154628a125fd2e8 👍
ddd8b0e5a3b35c4 1bb8db16cda874ff52af40a9ffdf56d7510abed65a9dec69f ⚠️ index [13]
⚠️ J Jean-Marc, machine id 04b1f154204fa9e9:
👉🏼 The source or parameters have been modified or monkey patched
⚠️ index 54 hash: d5aa069407214f52b187ba479047d36fead12bc5b541ba3bceece6bc9f328490
⚠️ L Bobby, machine id 04b1f154204fa9e9:
    👉🏼 The source or parameters have been modified or monkey patched
hash found:
880414219b2eadf47f234227c28927fc7d8a6f1a911bf40fa023f58d1a6cd83d ⚠️ index [43]
7de9df303f6cbbd183bcbc3745f4c2ed87d4e9a38afc79f8a47ea47981de43a5 ⚠️ index [45]
d5aa069407214f52b187ba479047d36fead12bc5b541ba3bceece6bc9f328490 ⚠️ index [50]
⚠️ Lamarq Linda, machine id 04b1f154204fa9e9:
    
👉🏼 The source or parameters have been modified or monkey patched
hash found:
fb85b7bceb28ecce4a047d1fb94428c60789505ebd89e22a5154628a125fd2e8 👍
4105fbe50a65562da899d9d9062a1853f8623c6b14b7f0a663cfc5244afd0b40 ⚠️ index [30]
⚠️ Morane Bob, machine id 04b1f154204fa9e9:
    👉🏼 The source or parameters have been modified or monkey patched
⚠️ index 36 hash: b78c45342e260e62883a229b0b48364956e78b039ddbd0d64b6 222d761ae8e4a
```
## Client-side security
As mentioned above, if you want to grade students or conduct an exam under proper conditions, we recommend using a question file WITHOUT answers. Since the Python source code is accessible, a student or group of students could study it, reverse engineer it, and tamper with the data sent to the server. This is reasonably complex and difficult, but it cannot be ruled out.
 
Therefore, "client-side" security should be added with an unpredictable intervention. This could be a supervisor walking among the rows, or the execution on demand (by the teacher or students) of a specific hidden code protected by a password that is only revealed at the time of execution. This has been implemented and should significantly reduce the temptation to cheat. 
The idea is to prepare a code, which will be encrypted with a password, executed on demand, and which will perform a number of integrity checks on the student's configuration.
The teacher prepares their encrypted archive as follows:
```python
# example_create _secure_tar.py
# ⚠️ zutils.py must be present in the local directory
from labquizdev.putils import create_secure_tar
# Install python_minifier if necessary
import importlib.util
import sys
import subprocess
module_name = "python_minifier"
  
spec = importlib.util.find_spec(module_name)
if spec is None:
    print(f"{module_name} is not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", ‘pip’, "install", module_name])
import python_minifier
try:
    with open(‘zutils.py’ , ‘r’) as f:
            minified = python_minifier.minify(f.read(), remove_literal_statements=True, 
                            rename_globals=True, preserve_globals="do_the_check")
    with open(‘quiz_data/zutils.py’, ‘w’) as f:
            f.write(minified)
except:
    pass
        
global_hash, src_hash = create_secure_tar(
    source_dir="quiz_data",
    output_file="quiz.tar.enc",
    password_open="PASSWORD_TO_ENCRYPT__THE__ARCHIVE",
    
password_seal="SECOND_PASSWORD_TO_WATERMARK_SOURCES"
)
```
:::{figure} doc_images/secure_tar.png
:name: secure_tar
:label:fig17
:alt: Create secure encrypted tar
:align: center
:width: 60%
Execution result
:::
Inside this archive, the zutils module is completely autonomous and contains a customizable `do_the_check` function that is launched automatically and performs checks that are transmitted to the server. In the version provided to teachers (but customizable, `extras` directory), a *daemon* is launched that performs the check periodically. It contains the integrity code of the zutils.py sources, watermarked, under the `session_hash` key.
On the student side, simply call
```
quiz.check_quiz()
```
on each machine, using the `password_open` password.
:::{figure} doc_images/check_quiz.png
:name: check_quiz
:label:fig18
:alt: Check config integrity
:align: center
:width: 60%
Execution result
:::

---
title: LabQuiz
subtitle: A suite of tools for integrating quizzes into Jupyter notebooks
date: 02/09/2026
license: CC-BY-NC-SA-4.0
authors:
  - name: Jean-François Bercher
    email: jf.bercher@esiee.fr
    url: https://perso.esiee.fr/~bercherj/
    corresponding: true
    orcid: 0009-0007-5474-7475
    affiliations:
      - name: LIGM
        url: https://ligm.univ-eiffel.fr/
      - name: Dept IT, ESIEE
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
numbering:
  title: 
    enabled: false
    offset: 0
  heading_1: true
  heading_2: true
  heading_3: true
math:
  '\sha': 'ш'
  '\dr': '\mathrm{d}#1'
  '\wb': '\mathbf{w}'
  '\Db': '\mathbf{D}'
  '\Kb': '\mathbf{K}'

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
(installation)=
# Installation
🆙 🛠  For the moment, and before any distribution on pypi, to install LabQuiz on your system, consult
[this repository](https://perso.esiee.fr/~bercherj/JliteNotes/pypi/), note the highest version number x.y.z and 
enter
```{code}
pip install https://perso.esiee.fr/~bercherj/JliteNotes/pypi/labquiz-x.y.z-py3-none-any.whl --force-reinstall
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
                            
:::{figure} doc_images/login.png
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
 
:::{figure} doc_images/4boutons_submit_actif.png
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
:::{figure} doc_images/2boutons_submit_actif.png
:name:
 
quiz3
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
(question_file_structure)=
# Question file structure - 🧑‍🏫🏫
## General structure
The question file is a [YAML](https://en.wikipedia.org/wiki/YAML) file. The format is simple, but the number of spaces or indentations must be consistent throughout the file. 

The general structure is as follows:

```{code}
QuizFile
│
├── title (string) [optional]
│
└── quiz_id (ex: quiz1, quiz23, ...)
    │
    ├── question (string)
    ├── type (mcq | numeric | mcq-template | numeric-template) [optional]
    ├── label (string)
    │
    ├── constraints [optional]
    │     └── list
    │           └── constraint
    │                 ├── indexes [list of labels]
    │                 ├── type (XOR | SAME | IMPLY | IMPLYFALSE)
    │                 └── malus [optional]
    │
    ├── variables [templates only]
    │     └── variable_name
    │           ├── type
    │           ├── structure
    │           ├── engine
    │           └── call
    │
    └── propositions (list)
          │
          └── proposition
                ├── proposition (string)
                ├── label [optional]
                ├── type (bool | float | int | numeric) [optional]
                ├── expected (any)
                ├── tolerance [numeric only]
                ├── tolerance_abs [numeric only]
                ├── tip [optional]
                ├── answer [optional]
                ├── bonus [optional]
                └── malus [optional]
```

with the following relationships
```{mermaid}

classDiagram

class QuizFile {
    +title : string (optional)
    +quiz_id : Quiz (1..*)
}

class Quiz {
    +question : string (REQUIRED)
    +label : string (REQUIRED)
    +type : string = "mcq" (optional)
    +propositions : Proposition[1..*] (REQUIRED)
    +constraints : Constraint[] (optional)
    +variables : Variable[] (templates only)
}

class Proposition {
    +proposition : string (REQUIRED)
    +expected : any (REQUIRED)
    +label : string (recommandé)
    +type : bool|float|int|numeric (optional)
    +tolerance : float (numeric only)
    +tolerance_abs : float (numeric only)
    +tip : string (optional)
    +answer : string (optional)
    +bonus : int (optional)
    +malus : int (optional)
}

class Constraint {
    +indexes : string[] (REQUIRED)
    +type : XOR|SAME|IMPLY|IMPLYFALSE (REQUIRED)
    +malus : int (optional)
}

class Variable {
    +type : string (REQUIRED)
    +structure : string (REQUIRED)
    +engine : string (REQUIRED)
    +call : string (REQUIRED)
}

QuizFile --> Quiz
Quiz --> Proposition
Quiz --> Constraint
Quiz --> Variable

```


The file begins with a line
```
title: an explanation of the file's contents
```
It then contains a list of quizzes, each beginning with a label, for example (but there are no restrictions on the choice of labels
```
quiz1:
    ...
quiz2:
    ...
    
```
Each quiz itself contains a question, followed by a series of options. 
- There is no limit to the number of options; there must be at least one.
  
- The texts of the questions and answers are character strings, which can be enclosed in single or double quotation marks. These quotation marks are not mandatory, unless the string contains a :, in which case you must enclose it in single quotation marks (and in this case, if the string also contains single quotation marks, they must be doubled, see the examples below)
```
quiz23:
    question: This is the text of the question
      ...
    propositions:
      - proposition: text of the first proposition
        ...
      - proposition: 'text of the second proposition with a : that must be taken into account'
        ...
    
```
This is the bare minimum.
By default, the quiz is of the "mcq" type. It can also be of the "numeric" type, in which case this must be specified. If you want to be able to use online correction and display tips, you will need to add them.

(structure_type_qcm)=
## Multiple choice type
```
quiz23:
    question: This is the text of the question
    type: "mcq"         #"mcq" or "numeric" (optional - "mcq" by default)
    propositions:
        - proposition: text of the first option (incorrect)
          type: bool      # optional - default "bool" if type "mcq", ‘float’ if type "numeric"
          label: label1   # optional but necessary for corrections                
          expected: false # expected value for the answer
          tip: text of a clue or hint towards the correct answer
          answer: explanatory text for the correct answer
        - proposition: ‘text of the second proposition (true) with a : that must be taken into account’
          label: label2
          type: bool
          expected: true
          tip: text of a clue or guidance towards the correct answer, with quotes '' if necessary
          answer: explanatory text for the correct answer, with quotes ‘’ if necessary
```
- *Consistency constraints* on propositions can be added. For example, it can be required that the true answer to the proposition of label `label2` implies that the answer to the proposition `label1` is false. In case of violation, a penalty is applied.
- Similarly, certain propositions can give rise to a *bonus* or a *penalty*. The bonus is the number of points awarded if the answer is the expected one (default 1) and the penalty is the number of points deducted if the answer is different from the expected one (default 0). 
With these elements, the example could be completed as shown below. The implementation is then given [](fig10).
 
```
quiz23:
    question: This is the text of the question
    type: "mcq"    #"mcq" or "numeric" (optional - default "mcq")
    constraints: [
          { "indexes": ["label2", "label1"], "type": ‘IMPLYFALSE’, "penalty": 2 }
          ]
    propositions:        
      - proposition: text of the first proposition (false)
        type: bool        # optional - default "bool" if type "qcm", ‘float’ if type "numeric"
        label: label1      # optional but necessary for corrections          
        expected: false    # expected value for the answer
        tip: text of a clue or hint towards the correct answer
        answer: explanatory text for the correct answer         
      - proposition: ‘text of the second proposition (true) with a : that must be taken into account’
        label: label2
        type: bool
        expected: true
        tip: text of a clue or guidance towards the correct answer, with quotes '' if necessary          
        answer: explanatory text for the correct answer, with quotes '' if necessary
      - proposition: text of a third proposition with penalty
        label: label3
        type: bool
        expected: true
        malus: 2     #penalty applied here if the box is not checked
        tip: text of a hint 
```


:::{figure} doc_images/quiz23.gif
:name: quiz23
:label:fig10
:alt: Quiz example
:align: center
:width: 60%
Implementation of question `quiz23`. Note that the propositions are automatically mixed.
 
:::
Several logical constraints can be specified, as in this example: 
```
quiz57:
  question: "This is a question with contradictions and implications. The number is 6"
  type: "mcq"
  constraints: [
      { "indexes": ["parity", "odd"], "type": "XOR", "penalty": 2 },
      { "indexes": ["parity", "multiple of 2"], "type": "SAME", "penalty": 2 },
      { "indexes": [‘parity’, "plus1pair"], "type": "XOR", "penalty": 2 },
      { "indexes": ["parity", "plus1odd"], "type": ‘IMPLY’, "penalty": 2 }
    ]
```

## Type `numeric`
For questions with numerical values, the pattern is similar. Additional keys are available: `tolerance`, `tolerance_abs`
- `tolerance` is the percentage of variation tolerated on the expected value
- `tolerance_abs` is the absolute tolerance.
 
The tolerance used during correction is the greater of the values between tolerance_abs and tolerance*expected.
The `type` in each proposition can be `float` (default) or `int`.
Bonuses (default 1) and penalties (default 0) can also be specified and are applied depending on whether the difference between the given value and the expected value is greater or less than the tolerance.
 
```
quiz24:
  question: Please enter below the number of points and the values of the mean and standard deviation of the time series
  type: numeric
  propositions:
    - proposition: Mean:
      label: mean
      type: float
      expected: 0.0
      answer: 0
      tolerance: 0.05
      tolerance_abs: 0.01
      tip: Enter the value
    - proposition: Standard deviation
      label: sigma
      type: float
      expected: 1.0
      answer: "1"
      tolerance: 0.05
      tolerance_abs: 0.01        
      tip: Enter the value
    - proposition: Number of points
      label: N
      type: int
      expected: 512   
      answer: The number of points `len(series)` or `series.shape[0]` is 512
      tolerance: 0.01
      tolerance_abs: 2
      bonus: 2
      penalty: 3
      tip: Enter the value
```
:::{figure} doc_images/quiz24.gif
:name: quiz24
:label:fig11
:alt: Quiz example
:align: center
:width: 60%
Implementation of the `quiz24` question of the numeric type. Note that the propositions are automatically mixed and that a bonus/penalty is applied to the number of points.
 
:::
## Case of `templates`
Two additional types are possible, namely `numeric-template` and `qcm-template`. These formats allow the use of variable numerical data and values related to the experiments carried out in the practical. This data is passed as a parameter to the `show()` function and used for correction. For multiple-choice questions, it is possible to test whether the result belongs to an interval or other calculable conditions whose result is Boolean. For correction, a formula is implemented that calculates the expected value based on the parameters passed.
 
In the following example, the `show` function is called with two parameters:
```
quiz.show("quiz54", a=res1, b=res2)
```
These two parameters are used to calculate the solution. For example, the formula `f'{a+b:.4f} '` (invisible to the student!) is *evaluated* with the context that is 
passed to the function, and this context is simultaneously *saved* on the remote server to allow the teacher to recalculate the solution offline. The formula can also be specified naturally as "a+b" or as "{a + b}", with or without quote or brace. 

A field `variables` can alse be precised in the quiz file, as
```
variables:
  a:
    type: int
    structure: scalar
    engine: numpy rng.
    call: integers(0, 10, size=1)
  b:
    type: int
    structure: scalar
    engine: numpy rng.
    call: integers(0, 10, size=1)
```
This enables to provide values for the different variables, which is useful for export to HTML or AMC-$\LaTeX$ format,  but this also allows values to be generated on the fly if they are not given in the quiz. The `autovars=True` option is provided for this purpose, e.g.
```
quiz.show("quiz54", autovars=True)
```

```
quiz54:
  question: "This is a numerical question where you have to calculate the sum and difference of {a} and {b}"
  type: "numeric-template"
  variables:
    a:
      type: int
      structure: scalar
      engine: numpy rng.
      call: integers(0, 10, size=1)
    b:
      type: int
      structure: scalar
      engine: numpy rng.
      call: integers(0, 10, size=1)
  propositions:
    - proposition: "Sum: "
      label: 'sum'
      type: "float"      
      expected: f'{a+b:.4f}'
      answer: The sum of {a} and {b} is {a+b}
      tolerance: 0.01
      tolerance_abs: 0.01      
      tip: "Enter the value"
    - proposition: "Difference: "
      label: ‘difference’
      type: "float"
      expected: f'{a-b:.4f}'
      answer: The difference is {a-b}
      tolerance: 0.01      
      tolerance_abs: 0.01
      tip: "Enter the value"
```
:::{figure} doc_images/quiz54.png
:name: quiz54
:label:fig12
:alt: Quiz example
:align: center
:width: 60%
Implementation of the `quiz54` question of numeric type. Parameters are passed to the function, which evaluates and calculates the correct solution based on these parameters.
:::
Almost any Python function can be used to evaluate the answer.
 
Just keep in mind that the context is saved in the remote spreadsheet. Therefore, avoid making it too large: avoid large data tables! and prefer smaller contexts. Not all contexts and data types are necessarily serializable (dictionaries, lists, numpy arrays, and pandas are supported here). In addition, to recalculate the solution offline, you must store the useful modules.
 
In the following example, we calculate the coefficient of variation of a series using numpy. The call would be, for example
```
quiz.show("quiz61", s=np.random.randn(5), np=np)
```
where we pass the name of the module(s) used in the context.
```
quiz61:
  question: "This is a numerical question where you have to calculate the coefficient of variation"
  type: "numeric-template"
  propositions:    
    - proposition: "Coefficient of variation"
      label: ‘cv’
      type: "float"
      expected:  np.std(s)/np.mean(s)
```
:::{figure} doc_images/quiz61.png
:name: quiz61
:label:fig13
:alt: Quiz example
:align: center
:width: 60%
Implementation of the `quiz61` numeric question. Parameters are passed to the function, including a module that evaluates and calculates the correct solution based on these parameters. 
:::


(score_calculation_correction)=
# Score calculation and correction by the teacher
(score_calculation)=
## Score calculation
A score is calculated automatically for each quiz and displayed in learning and test modes.
 
An overall score, which is the average of the results on the quizzes taken, can be accessed via `quiz.score_global`. The score is calculated using the two functions `calculate_quiz_score` and `correct_ans`, which are available in the `utils` and `putils` modules, respectively. The main principles are as follows:
- The function receives the user's answer user, a weight matrix, and the constraints imposed
- **weight matrix**  - the *weight matrix* is used to weight the answers given according to the expected answer: number of points awarded if the user answered true when the expected answer was true, if they checked true when the expected answer was false, etc. The default matrix is as follows: 
```
        default_weights = {
                (True, True):   1,  # True Positive  # Checked and the correct answer was true
                (True, False): -1,  # False Positive  # Checked when the correct answer was false
                (False, True):  0,  # False Negative  # Did not check though the correct answer was true
                (False, False): 0   # True Negative  # Did not check and the correct answer was indeed false
            }
```
In short, here we award one point for correct answers that were checked and deduct one point for correct answers that were not checked. Depending on the case, we might want to adapt this to penalize correct answers that were not checked, or give less weight to an omission, for example
```
        
weights = {
                (True, True):   1,   # True Positive    # Checked and the correct answer was true
                (True, False): -0.5, # False Positive    # Checked when the correct answer was false
                (False, True): -0.5, # False Negative    # Not checked when the correct answer was true
                (False, False): 0    # True Negative    # Not checked and the correct answer was indeed false
}
```
We can also impose an *identity* weight, which means that we award 1 point for each true statement checked and 1 point for each false statement not checked. A disadvantage of this strategy is that, obviously, a blank copy with no checks gets statistically half the points, if there are as many wrong statements as right ones, or even more if it is a "single correct answer" questionnaire. 
```
weights = {
                (True, True):   1,   # True Positive    # Checked and the correct answer was true
                (True, False): 0, # False Positive    # Checked when the correct answer was false
                (False, True): 0, # False Negative    # Did not check when the correct answer was true
                (False, False): 1    # True Negative    # Did not check and the correct answer was indeed false
}
```

The weight matrix cannot be modified when calculating the score online, but it can be modified when recalculating retrospectively based on the recorded results; see correction by the teacher [](correction_by_teacher)
- **bonus malus** - As we have seen in the file structure [](question_file_structure), `bonuses` and `penalties` can be integrated into the questions themselves. This allows, for a given question, to give more weight to a particular correct answer, or conversely to penalize a wrong answer, independently of the general weighting matrix. 
- **logical constraints** - *Logical constraints* can be integrated into questions and used to calculate the score. These constraints are specified question by question in the question file, and penalties are applied if the constraint is violated. The following constraints can be used: 
```
    # constraints: List of dictionaries e.g.: [{"indexes" : (0, 1), "type": ‘XOR’, "penalty": 2}]
    - XOR (Exclusion) A and B must be different
    - IMPLY (Implication) If A is TRUE, then B is TRUE
    - SAME (Consistency) A and B are equivalent (same value)
    - IMPLYFALSE: If A is TRUE, then B MUST be FALSE
```
See an example at the end of [](structure_type_qcm).
- **numerical values** - In the case of numerical answers, the difference between the given value and the expected value is calculated. If this difference is less than the threshold defined by the tolerance, the answer is counted as correct[^1].
Remember that the tolerance is specified in the question file, and that the greater of the values between tolerance_abs and tolerance_relative*expected is used. If the relative tolerance `tolerance` has not been specified, the value used is 1%. 
Bonuses (default 1) and penalties (default 0) may also be  applied depending on whether the difference between the given value and the expected value is greater or less than the tolerance.
 
[^1]: It would be possible to set the score based on the (relative) value of this difference, but this has not been done and will be discussed later.  

(teacher_correction)=
## Correction by the teacher
In parallel with the practical work, or *a posteriori*, the teacher can use the automatically recorded results to monitor progress, adjust the weights (weight matrix) or the grading scale. Obviously, this is essential in `exam mode`, where students do not receive feedback on their answers and where the corrections are not accessible. 
To correct all recorded answers, the correction is as simple as:
```python
from labquiz.putils import correctQuizzes
URL = "https:// URL_USED_TO_COLLECT_RESULTS"
SECRET = "SECRET_PASSWORD_SPECIFIED_IN_THE_SHEET"
QUIZFILE = "NAME_OF_QUESTION_FILE.yml" #quiz file CONTAINING the expected values
#
Res = correctQuizzes(URL, SECRET, QUIZFILE)
```
This provides a table of results in the following form
```
Res
```
:::{figure} doc_images/results_table.png
:name: results
:label:fig14
:alt: Results table
:align: center
:width: 60%
Example of a results table.
:::
which can of course be exported, for example with
```python
Res.to_csv("Results.csv")
```
## Correction for an exam generated with `quiz.exam_show()`
For an exam generated with `quiz.exam_show()`, which is identified by the title given to it when it was created, the teacher can specifically correct the data collected on the server, according to
(after retrieving the data with `readData` and instantiating a quiz -- steps 1 and 3 above),
```python
from labquiz.putils import getExamQuestions, getAllStudentsAnswers, correctAll
exam_questions = getExamQuestions("Test to see", data)
students = exam_questions.keys()
students_answers = getAllStudentsAnswers(students, data, maxtries=3)
correctAll (students_answers, quiz, data_filt, threshold=0, 
           exam_questions=exam_questions, weights=None, grading=None, maxtries=3)
```
```python
from labquiz.putils import correctQuizzes
URL = "https://URL_UTILISÉE_POUR_RECUEILLIR_LES_RÉSULTATS"
SECRET = ‘SECRET_PASSWORD_SPECIFIED_IN_THE_SHEET’
QUIZFILE = " NAME_OF_QUESTION_FILE.yml" #quiz file CONTAINING the expected values
Res = correctQuizzes(URL, SECRET, QUIZFILE, title=‘Test title’)
```
where the title of the test to be corrected is specified by the `title` parameter.
:::{figure} doc_images/exam_show_results_table.png
:name: results
:label:fig15
:alt: Results table
:align: center
:width: 80%
Example of a results table (`exam_show`). 
:::
## Correction options
A few additional options can be used during correction. 
```python
def correctQuizzes(URL, SECRET, QUIZFILE, title=None, threshold=0, weights=None, grading=None, maxtries=1)
````
- title: if title is not None, this indicates that it is the correction of a test with randomly selected questions of type `exam_show`, and whose title is title,
- threshold: threshold=0 sets the scores for each question to zero (otherwise negative scores are possible)
 ; this is the default value, but if you want to allow negative scores per question, you can set it to -10, for example.
- weights: the weight matrix (dictionary) already discussed in [](score_calculation)
- scale:  weight of the different questions in the quiz. If there is no scale, all 
            questions have the same weight for the calculation of the score. If the weight of a question 
            is not specified, it defaults to 1. Example: bareme = {‘quiz3’:4, ‘quiz55’:0} assigns a coefficient of 4 to question quiz3 and neutralizes question quiz55 (all other questions will have a weight of 1),
- maxtries: Number of attempts allowed. Correction is performed on the last attempt less than or equal to maxtries (and before any correction request, "Correct" button, if available).

## Dashboard
During the session, you can retrieve the data table (see previous section) and view the results. The following basic functions allow you to monitor overall progress
```python
# Reading data
URL = "https://URL_UTILISÉE_POUR_RECUEILLIR_LES_RÉSULTATS"
SECRET = "SECRET_PASSWORD_SPECIFIED_IN_THE_SHEET"
data, data_filt = readData(URL, SECRET)
df_last = data_filt.drop_duplicates(
    subset=["student", ‘quiz_title’],
    keep="last")
#The number of quizzes completed per student
quiz_count_by_student = (
    df_last.groupby("student")["quiz_title"]
           .apply(len)
           .sort_index(ascending=False)
)
# Scores per student
score_by_student = (
    df_last.groupby("student")[‘score’]
           .apply(np.sum)
           .sort_index(ascending=False)
)
# For the entire class
class_counts = (
    df_last["quiz_title"]
    .value_counts()
    .sort_index (
        key=lambda idx: idx.str.extract(r"(\d+)").astype(int)[0]
    )   # alphabetical order
)
#
print("Number of quizzes completed\n\n", quiz_count_by_student)
print("Scores obtained\n\n", score_by_student)
print("Complete class\n\n", class_counts)
```
We can even put this in the form of a small graphical *dashboard* that we will refresh regularly.
The dashboard mentioned above was finally formatted into a dedicated application, which is described in [](quiz_dash) . 

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
(prepa_encode_file)=
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
`quiz_editor` -- editing a suggestion -- correct or incorrect, hint (tip), answer (displayed during correction), bonus, penalty, etc.
:::
:::{figure} doc_images/exports.png
:name: quiz_editor_exports
:label:fig26
:alt: quiz_editor_exports
:align: center

`quiz_editor` -- Export examples (YAML, AMC-$\LaTeX$, interactive HTML, HTML-exam)
:::

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
(google-sheet)=
#  Create your Google Sheet to collect results - 🧑‍🏫🏫
To collect the data, you can simply use a Google Sheet. In a later version, a small Flask server could be used, but it would need to be deployed somewhere. For now, a Google Sheet version is functional and easy to implement.
## The simplest of the simplest...
```{hint}
 
Hint:  The simplest of the simplest
**The simplest of the simplest** is to use a copy of the template that has been prepared for this purpose:
To do this, click on the [following link]( 
https://docs.google.com/spreadsheets/d/1-hDtosDAA3ehy4iqFGU5D5NQirGY3c-HxLDmnsoG6AU/edit?usp=sharing)
- then go to the File menu/Create a copy (rename the file),
- and deploy via: Extensions/Apps Script menu and click deploy in the top right corner; choose `Web Application`, share with everyone.
Copy and save the link `https://script.google.com/macros/ .../exec`, this is the URL you will use later (to share with students and use to read collected data). You can find this link by going to "Manage your deployments."
 
🎉 Save and you're done!
You can browse the following if you want to create the Google Sheet yourself or understand the meaning of the parameters in the Config sheet. You need to adjust the Pwd values in the Config sheet and the SECRET value in the AppScript code (Extensions/Deploy, 1st line). See points 4, 5, and 6 below.
 
```
- - -
## Manual preparation
- 1 - Create a Google Sheet using your account
- 2 - On the first line of the first sheet, insert the following data, which will serve as the header for your table:
```
timestamp send_timestamp  notebook_id student quiz_title  event_type  parameters  answers score
```
:::{figure} doc_images/1stRowSheet.png
:name: dashboard
:label:fig16
:alt: 1stRowSheet.png
:align: center
:width: 90%
First row of the sheet.
:::
- 3 - Create a new sheet by pressing `+` and rename it Config (by CTRL + clicking on the corresponding tab).
:::{figure} doc_images/creationfeuilleConfig.png
:name: dashboard
:label:fig17
:alt: creationfeuilleConfig
:align: center
:width: 30%
Creation of the `Config` sheet.
:::
- 4 - In the `Config` sheet, create the following data
```
Pwd Data reception
Wrktz TRUE
NMAX  NKEEP
2000  2500
````
:::{figure} doc_images/contenuFeuilleConfig.png
:name: dashboard
:label:fig18
:alt: contenuFeuilleConfig
:align: center
:width: 30%
Data from the `Config` sheet.
:::
👉🏼 Pwd is the key that will be used to check the correct connection with the sheet and participate in the encryption of the question file. Change the value of the key in A2 and keep it in mind.
 
NKEEP is the maximum number of rows kept in the sheet. Thresholding occurs as soon as the number of rows exceeds NMAX. 
👉🏼  In the Config sheet, select cell `B2`. Go to the `Data` menu and select `Data Validation`. Then, in `Data Validation Rule`, select `Checkbox`. Make sure the box is checked, otherwise you will not receive anything! 
⚠️ *And so to stop receiving data, uncheck*! --> This can be useful because some people leave things running on their computers, and since there is a `check_alive` integrity check sent periodically, this can fill up the Google Sheet (even though we have set a maximum limit). 
- 5 - In the `Extensions` menu, click on `Apps Script`. In the tab that opens, name your project, then in the code page, after deleting what is there, paste the code attached in the code_gs.txt file (`extras` folder) and replace the value of `SECRET` on the first line. If your first sheet is not called `Sheet 1`, rename it or change the constant `SHEET1`. txt file (`extras` folder) and replace the value of `SECRET` on the first line. If your first sheet is not called `Sheet 1`, rename it or modify the constant on the third line of the code. 
:::{figure} doc_images/ExtensionsAppsScript.png
:name: dashboard
:label:fig19
:alt: ExtensionsAppsScript
:align: center
:width: 60%
Create the extension to enter the Google Script code.
:::
- 6 - At the top right, click on `Deploy` then `New deployment`, choose the `Web application` type, share with everyone, copy the link `https: //script.google.com/...`. This is the URL you will use later (to share with students and to read the collected data). You can find this address by going to "Manage your deployments." 
🎉 Save and you're done!

:::{figure} doc_images/deployments.png
:name: dashboard
:label:fig20
:alt: manageDeployments
:align: center
:width: 60%
Deploy!
:::

(jupyterlite)=
# Deploy to Jupyterlite - 🧑‍🏫🏫
[Jupyterlite](https://jupyterlite.readthedocs.io/en/stable/index.html) is a Python/ Jupyterlab distribution that runs entirely in the browser. The various Python packages have been ported to WebAssembly (WASM), a binary format that is executable in browsers. For optimized libraries (numpy, pandas, etc.), performance is equivalent to native performance. For pure Python (loops, etc.), execution is 3 to 10 times slower. What this means, however, is that you can prepare Python programs and Jupyter notebooks that are fully executable in a browser, with zero installation! And it works independently of the system, even on tablets, phones, etc.
For example, you can find here: [https://perso.esiee.fr/~bercherj/JliteNotes/lab/](https://perso.esiee.fr/~bercherj/JliteNotes/lab/) an example of a practical assignment distributed in this way, incorporating labquiz. 
Preparation and deployment under Jupyterlite are not exactly immediate (there are documents to follow and adaptations to make), so I suggest a shortcut, which is as follows:
1. Download the archive of a deployed version, available here <br>
https://drive.google.com/file/d/14BnzVmPO6I8uOMEmNC6BTMsjXL3RLrQL/view?usp=sharing
<br> and unzip it on your disk,
 2. Place the files you want to make available in `_output/files`
3. Run the program `update_jupyterlite_contents.py` available in the `extras` directory by typing `python update_jupyterlite_contents.py`. Install dependencies if necessary. 
> 👉🏼 This will update the lists and characteristics of the files in the distribution
1. **init**) Upload *all* of the contents of `_output` to your web account, typically (or for instance...) to `login/public_html`, in a directory such as `MyJlite`, and you will obtain a distribution that is accessible and executable at `https://www.esiee.fr/~login/MyJlite`. If you don't know how to do this, ask your nearest local IT guru. 
2. **update**) If you only need to perform an update, simply copy the contents of `_output/files` **and** `_output/api` to `login/public_html/MyJlite/files` and `login/public_html/MyJlite/api`. 
For subsequent updates, add or modify files, you only need to perform steps 2, 3, and 5.maj); this will save you from having to upload the 70 MB in step 4.init). 

# Demonstration and exercises
Explore the demo notebook `labQuizDemo.ipynb` available in the `extras` directory and experiment. You can also use the jupyterlite version available [here](https://perso.esiee.fr/~bercherj/labquizDemo/lab/index.html?path=labQuizDemo.ipynb)
 
If you want to go further, and if you've made it this far, you're probably motivated, so here are a few ideas.
  

- **Exercise**: Create a Google Sheet named `MyFirstGSQuiz` by following the instructions in [](google-sheet).  Change the read password in the first line of the script accessible via Extensions/Apps Script, then deploy. Note the deployment address and password.
- * *Exercise**: Load or create a question database in `quiz_editor`. Add a few questions. Save the new file. Export an HTML version and an AMC version. Upload the HTML to your account and test (or test locally).
- **Exercise**: Export an encrypted version of your questions (note the password and update it in the Google Sheet, Config page) .
- **Exercise**: Create a Jupyter notebook, import labquiz, instantiate a quiz, -- see [](usage), integrate a few questions and test.
- **Exercise**: Instantiate a quiz linked to the URL of your Google Sheet, integrate a few questions into your notebook, and check that the answers appear in the Google Sheet.
 - **Exercise**: Use `quiz_dash` to read the results. Change the settings in your notebook (change `mode_examen` from True to False, modify `retries`), and see if this triggers an alert. Run the correction on your few tests. 
- **Exercise**: Create a jupyterlite version as explained in [](jupyterlite), add your files, upload it to your info account in `public_html`, and check by connecting to the corresponding web address.
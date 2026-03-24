---
title: Question file structure
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

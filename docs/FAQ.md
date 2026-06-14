---
numbering: false
---

# General

::::{dropdown} Short FAQ
#### **General Overview**

*   **What is LabQuiz?**
    LabQuiz is a comprehensive **suite of tools designed to integrate interactive quizzes directly into Jupyter notebooks**, primarily for educational purposes and practical assignments. It consists in `labquiz`, the primary package to insert and uses quizzes in the notebook, and two optional tools designed to edit quiz questions (`quiz_editor`) or monotor quiz sessions (`quiz_dash`) 
*   **Can students use LabQuiz without installing Python?**
    Yes. By deploying quizzes via **Jupyterlite**, the distribution runs entirely in the web browser using WebAssembly (WASM). This allows for a **"zero installation" experience** that works on computers, tablets, and phones.
*   **How students can benefit of labquiz?**
    Students benefit from the LabQuiz suite through its flexible access, immediate pedagogical feedback, and the ability to engage in personalized, interactive learning environments directly within their browser or Jupyter notebooks.
*   **What a student shall install?** 
    Students just need to install the labquiz package `pip install -U labquiz` as they usually do not need to edit questions (quiz_editor) nor to monitor a quiz session (quiz_dash)

#### **Question Creation & Structure**
*   **How do I format a question file?**
    Question files use the **YAML format**. They require strict and consistent indentation throughout the document. Each file generally starts with an optional title and contains a list of quizzes identified by unique labels (e.g., `quiz1`, `quiz23`).
*   **What types of questions are supported?**
    LabQuiz supports several types: **`mcq`** (multiple-choice, the default), **`numeric`**, and **template-based versions** (`mcq-template` and `numeric-template`) which allow for dynamic variables and randomized data.
*   **Can I use variables in my questions?**
    Yes, through **template questions**. You can define a `variables` block using engines like `numpy rng.` to generate random values. These variables can then be used in the question text or for calculating the `expected` answer using Python formulas.

#### **Scoring & Correction**
*   **How is the score calculated for a QCM?**
    Scores are calculated using a **weight matrix**. By default, a "True Positive" (checking a correct answer) earns 1 point, while a "False Positive" (checking an incorrect answer) deducts 1 point. "True Negatives" and "False Negatives" default to 0 points.
*   **Can teachers adjust the scoring system?**
    Yes. Teachers can implement **bonuses and penalties** for specific propositions within the YAML file. Additionally, using the **`quiz_dash`** tool, they can retrospectively adjust the weight matrix or the overall scoring scale for the entire questionnaire.
*   **How do I collect student results?**
    In connected mode, results are **automatically recorded on a remote server** (often via a Google Sheet). Teachers can then use the `correctQuizzes` function or the `quiz_dash` interface to download a complete results table in CSV or PDF format.

#### **Security & Exams**
*   **Is LabQuiz secure for high-stakes exams?**
    The suite includes several security layers, such as **machine identification**, mandatory internet mode, and **hashing of source code** to detect "monkey patching" or parameter tampering.
*   **What is the most important security tip for teachers?**
    To conduct an exam under optimal conditions, you should **distribute a version of the YAML file that does NOT contain the answers** (`expected` values). This prevents motivated students from reverse-engineering the file to find the correct responses.
*   **How can I detect if students are sharing computers?**
    The system tracks a unique machine ID for every transaction. Tools like `check_machine` or the `quiz_dash` integrity tab can **detect if the same machine ID is associated with multiple student names**.

#### **Available Tools**
*   **What is the purpose of the `quiz_editor`?**
    The `quiz_editor` is a graphical web app that allows you to **edit questions with Markdown support**, define variable generation for templates, and export quizzes to various formats like **AMC-LaTeX** (for paper exams) or **HTML**.
*   **What does `quiz_dash` do?**
    `quiz_dash` is a **monitoring dashboard** for teachers. It allows for real-time tracking of student progress, integrity checks (verifying that students haven't changed the number of retries), and the generation of detailed grading reports.

::::

::::{dropdown} How students benefit from the LabQuiz suite?

Students benefit from the LabQuiz suite through its flexible access, immediate pedagogical feedback, and the ability to engage in personalized, interactive learning environments directly within their browser or Jupyter notebooks.

### **1. Universal Accessibility and Convenience**
One of the primary benefits for students is the "zero installation" requirement when quizzes are deployed via **JupyterLite**. 
*   **Platform Independence:** Quizzes run entirely in the web browser, making them accessible on **computers, tablets, and phones** regardless of the operating system.
*   **Ease of Use:** Students can open a simple link to a practical assignment and immediately begin working without configuring complex Python environments.

### **2. Immediate Feedback and Learning Support**
LabQuiz is designed to act as a learning tool by providing real-time guidance during exercises:
*   **Instant Scoring:** In learning and test modes, students receive an **automatic score** immediately after submitting their answers, allowing them to gauge their understanding of the material instantly.
*   **Hints and Clues:** Question files can include **"tips"**—clues or guidance that students can access if they are struggling with a specific proposition.
*   **Explanatory Answers:** Beyond just seeing if they were right or wrong, students can view **detailed explanatory text** for the correct answers, which helps clarify difficult concepts.
*   **Numerical Tolerance:** For math-based questions, the system can be configured with **tolerance margins** (percentage-based or absolute), ensuring that minor rounding differences do not unfairly penalize a student’s correct methodology.

### **3. Personalized and Varied Practice**
The use of **template questions** ensures that students have a unique and varied learning experience:
*   **Unique Data:** Templates allow for the generation of **dynamic variables**, meaning every student can receive different numerical values for the same problem, discouraging simple copying and encouraging genuine problem-solving.
*   **Self-Assessment Pages:** Students can use "Dynamic HTML" exports to access standalone **self-assessment web pages** where variables are regenerated every time the page is reset, providing endless practice opportunities.

### **4. Secure and Fair Evaluation**
For formal assessments, the suite ensures a professional and fair environment:
*   **Institutional Identity:** By integrating with **Google Workspace**, students can log in using their official university credentials, ensuring their work is correctly attributed and their data is secure.
*   **Detailed Reporting:** After an evaluation, instructors can generate and distribute **individual correction reports** in HTML or PDF format, providing students with a clear record of their performance and specific areas for improvement.
*   **Interactive Complexity:** The support for logical constraints (like XOR or implications) allows for more sophisticated questions that challenge students' critical thinking more effectively than standard multiple-choice tests.
  

::::


::::{dropdown} How do I install the LabQuiz suite locally via pip?

To install the various components of the LabQuiz suite locally, you can use the **pip** package manager for both the specialized graphical tools and their dependencies.

*   **Installing quiz_dash:** This real-time monitoring and analysis dashboard can be installed from PyPI by running **`pip install quiz-dash`**. Alternatively, it can be installed directly from the GitHub repository using **`pip install git+https://github.com/jfbercher/labquiz.git#subdirectory=quiz_dash`**. Once installed, you can launch it from the command line by simply invoking `quiz_dash`.
*   **Installing quiz_editor:** To install the graphical editor for preparing and encoding question files, use the command **`pip install quiz-editor`**. It is also available via git through the command **`pip install git+https://github.com/jfbercher/labquiz.git#subdirectory=quiz_editor`**. This tool can be started from the terminal using `quiz_editor [FILENAME.yaml]`.
*   **Installing Additional Dependencies:** 
    *   If you are preparing a **secure encrypted archive** (secure tar) for an exam, you may need to install the **`python_minifier`** package via **`pip install python_minifier`**.
    *   When deploying to **Jupyterlite**, you should run the `update_jupyterlite_contents.py` script located in the `extras` directory and **install any necessary dependencies** it requires to update the distribution's file characteristics.
    *   For **dynamic AMC-LaTeX exports**, the system requires the **`pythonTeX`** LaTeX package to be installed on your system to handle randomized variables.

::::

::::{dropdown} Can I run the labquiz on a phone or tablet?

Yes, you can run the quiz on a **phone or tablet**.

This is primarily achieved through **Jupyterlite**, which is a Python/JupyterLab distribution that runs entirely within a web browser. Because it uses WebAssembly (WASM) to execute code, it is independent of the operating system and requires **zero installation** on the device. It is specifically designed to be fully executable on **tablets, phones, and other mobile devices**.

In addition to Jupyterlite, the **quiz_editor** provides export options that are well-suited for mobile use:
*   **Dynamic HTML:** This export creates a "self-assessment" web page with integrated answers and correction mechanisms that can be accessed via a mobile browser.
*   **HTML Exam:** This creates a web-based exam interface that supports real-time submission of results to a Google Sheet, allowing students to participate using their mobile devices.

While the environment is fully compatible with mobile hardware, note that **pure Python execution** (such as complex loops) may be **3 to 10 times slower** in a browser-based environment compared to a native local installation, though performance for optimized libraries like *numpy* and *pandas* remains equivalent to native speeds.
::::


::::{dropdown} Give the list and signification of all parameters in QuizLab()

The `QuizLab()` class constructor accepts the following parameters to configure the behavior, security, and connectivity of a quiz session within a Jupyter notebook.

### **Core Identification & File Parameters**
*   **`URL`** (Default: `""`): The web address of the Google Sheet web application (the `/exec` link) or the remote server where student results and integrity data are recorded.
*   **`QUIZFILE`** (Default: `""`): The name or path of the **YAML file** containing the quiz questions, propositions, and (optionally) the answers.

### **Authentication & Connectivity Parameters**
*   **`needAuthentification`** (Default: `True`): A boolean that enables the requirement for participant identification. This is the standard authentication mode.
*   **`googleAuthentification`** (Default: `False`): A boolean that specifies the use of **institutional Google accounts** for login. When enabled, the quiz file is specifically encrypted to ensure it can only be decrypted by authenticated users.
*   **`mandatoryInternet`** (Default: `False`): A boolean that forces the quiz to run in **connected mode**. It is mandatory for Google authentication and ensures that all student validations and background integrity heartbeats are sent to the server in real-time.
*   **`CHECKALIVE`** (Default: `600`): An integer defining the interval (in seconds) at which the background **daemon** sends an integrity "heartbeat" to the server to check the system status.

### **Session & Integrity Parameters**
*   **`retries`** (Default: `2`): An integer defining the maximum number of attempts allowed for each quiz question.
*   **`exam_mode`** (Default: `False`): A boolean that flags the session as a formal exam. In this mode, correction buttons and tips are usually hidden, and results are recorded for later grading.
*   **`test_mode`** (Default: `False`): A boolean used to set the quiz to a practice or training state.
*   **`groups`** (Default: `[]`): A list of authorized student classes or groups (e.g., `['A', 'B', 'C']`) allowed to participate in the session.

### **Internal & Environment Parameters**
*   **`in_streamlit`** (Default: `False`): A boolean used to indicate if the quiz instance is running within a Streamlit environment rather than a standard Jupyter notebook.

### **Integrity Monitoring Note**
The suite's integrity system specifically monitors a **"watchlist"** of parameters to detect if students attempt to modify them via "monkey patching." This watchlist includes **`exam_mode`**, **`test_mode`**, and **`retries`**. Any deviation from the original values established during instantiation will be flagged by tools like `quiz_dash` or the `check_hash_integrity` function.

The `QuizLab()` class constructor accepts the following parameters to configure the behavior, security, and connectivity of a quiz session within a Jupyter notebook.

::::

::::{dropdown} How to deploy quizzes on Jupyterlite for installation-free execution?

To deploy quizzes on **Jupyterlite**, allowing for a "zero installation" experience where everything runs directly in the browser, you can follow a streamlined "shortcut" method described in the documentation.

### 1. Understanding Jupyterlite
Jupyterlite is a distribution of JupyterLab that runs entirely in the web browser using **WebAssembly (WASM)**. This means students can access and execute your quizzes on any device—including tablets and phones—without installing Python or any local software. While optimized libraries like *numpy* or *pandas* perform at native speeds, pure Python code may run **3 to 10 times slower**.

### 2. Deployment Steps (The Shortcut Method)
Because a full deployment from scratch can be complex, the following shortcut is recommended:

1.  **Download the Archive:** Obtain the pre-deployed version of the Jupyterlite archive (link provided in the documentation) and unzip it on your computer.
2.  **Add Your Content:** Place the YAML question files and any Jupyter notebooks you want to share into the directory named **`_output/files`**.
3.  **Update the Distribution:** Navigate to the `extras` directory and run the following command in your terminal:  
    `python update_jupyterlite_contents.py`  
    This script automatically updates the file lists and characteristics so the distribution recognizes your new content.
4.  **Initial Upload:** Upload the entire contents of the **`_output`** folder to your web hosting account (e.g., a `public_html/MyQuiz` directory).
5.  **Access the Quiz:** Once uploaded, the quiz will be accessible via a standard URL such as `https://your-server.fr/~login/MyQuiz`.

### 3. Managing Future Updates
If you need to add or modify files later, you do not need to re-upload the entire 70 MB distribution. Instead:
*   Modify the files in `_output/files`.
*   Run the `update_jupyterlite_contents.py` script again.
*   Copy only the **`_output/files`** and **`_output/api`** directories to your web server.

**Note on Security:** For exams or graded assignments, it is critical to distribute a version of your YAML question file **without the answers** (`expected` values), as a motivated student could otherwise reverse-engineer the file to find them.

### What is the download link for the Jupyterlite archive?

The download link for the pre-deployed Jupyterlite archive is:

**[https://drive.google.com/file/d/14BnzVmPO6I8uOMEmNC6BTMsjXL3RLrQL/view?usp=sharing](https://drive.google.com/file/d/14BnzVmPO6I8uOMEmNC6BTMsjXL3RLrQL/view?usp=sharing)**

This archive is provided as a **"shortcut"** for teachers and developers because a full preparation and deployment of Jupyterlite can be complex and time-consuming. By downloading and unzipping this version, you can quickly integrate your own quizzes and notebooks by placing them in the `_output/files` directory and running the `update_jupyterlite_contents.py` script found in the `extras` folder.

::::

::::{dropdown} How to structure a LabQuiz question file in YAML format?

Structuring a LabQuiz question file requires using the **YAML** format, which demands strict consistency in indentation and spacing throughout the document.

Here is a detailed guide on how to structure your file based on the technical documentation:

### 1. General File Structure
*   **Title (Optional):** You can begin the file with an overall title: `title: "My Quiz Collection"`.
*   **Quiz Identifiers:** Each question must start with a unique label (e.g., `quiz1:`, `quiz23:`).
*   **Question Text:** The `question` field contains the text of the problem. If the text contains a colon (`:`), it **must be enclosed in single quotes**.

### 2. Question Types
The behavior of the quiz is defined by the `type` field (which defaults to `"mcq"` if omitted):
*   **mcq:** Multiple choice questions where answers are boolean.
*   **numeric:** For questions requiring a numerical value.
*   **mcq-template / numeric-template:** Advanced types that use dynamic variables and Python formulas.

### 3. Proposition Configuration
Every quiz must have a list of `propositions`. Each entry in this list can include:
*   **proposition:** The text of the specific option.
*   **expected:** The correct answer (e.g., `true`/`false` for MCQ, or a number for numeric).
*   **label:** An internal ID for the option. While optional, it is **required** for logical constraints or specific teacher corrections.
*   **type:** Specifies the data type, such as `bool`, `float`, or `int`.
*   **tip / answer:** `tip` provides a hint during the quiz, while `answer` provides an explanation after the student validates their choice.
*   **bonus / malus:** Used to assign specific points or penalties to a particular option.

### 4. Advanced Features
*   **Logical Constraints:** You can define relationships between options using the `constraints` field. For instance, you can specify that if "Option A" is true, then "Option B" must be false using types like `XOR`, `SAME`, `IMPLY`, or `IMPLYFALSE`.
*   **Numerical Settings:** For `numeric` types, you can set `tolerance` (percentage) and `tolerance_abs` (absolute value) to define the accepted margin of error.
*   **Variables (Templates only):** Template questions include a `variables` field to generate random data. The `expected` field can then use Python f-strings (e.g., `expected: f'{a+b:.4f}'`) to calculate the answer dynamically.

### 5. Security Best Practices
For actual **exams**, it is highly recommended to provide students with a version of the YAML file that **does not contain the `expected` values**. Since Python source code is accessible in Jupyter, motivated students could potentially reverse-engineer the file to find the answers if they are included.

::::

::::{dropdown} How can I define variables for template questions?

To define variables for **template questions** (such as `numeric-template` or `qcm-template`), you can use a dedicated `variables` field within your YAML question file or pass them directly through Python code.


### 1. Defining Variables in the YAML File
In the question file, the `variables` field allows you to specify how data should be generated automatically. This is particularly useful for creating randomized questions for different students.

Each variable definition includes the following sub-fields:
*   **type:** The data type (e.g., `int`, `float`).
*   **structure:** The data organization (e.g., `scalar`).
*   **engine:** The library used for generation (e.g., `numpy rng.`).
*   **call:** The specific function to call (e.g., `integers(0, 10, size=1)`).

**Example Structure:**
```yaml
quiz54:
  question: "Calculate the sum of {a} and {b}"
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
    - proposition: "Sum"
      expected: f'{a+b:.4f}'
```


### 2. Activating Automatic Generation
To trigger the generation of these variables when displaying the quiz in a notebook, you must use the `autovars=True` option in the `show()` function:
`quiz.show("quiz54", autovars=True)`.

### 3. Passing External Variables via Code
Instead of defining them in YAML, you can pass variables directly from your Jupyter notebook's workspace to the quiz. This is ideal for quizzes based on experimental results obtained during a practical assignment.

You pass these as keyword arguments in the `show()` function:
`quiz.show("quiz54", a=res1, b=res2)`.

### 4. Using Variables in Content
Once defined or passed, these variables can be used in several places:
*   **Question text:** Use curly braces, such as `{a}` or `{b}`, which will be replaced by their values.
*   **Propositions and Answers:** You can use them in the `answer` field for explanations.
*   **Expected values:** For correction, use Python f-strings like `f'{a+b:.4f}'` or standard Python expressions like `np.std(s)/np.mean(s)`.

### 5. Important Considerations
*   **Formatting:** If you use variable formatting for display (e.g., `{a+b:3f}`), remember that the **expected value** must still be computed explicitly (e.g., `round(a+b,3)`) to ensure correct numerical matching.
*   **Serialization:** Avoid passing very large data structures (like massive dataframes) as variables, as this context is saved to a remote server for teacher correction.
*   **Tooling:** The `quiz_editor` web application provides a graphical interface to define these variable generations without manual YAML editing.

::::

::::{dropdown} How to define a random number generator for template questions in a LabQuiz YAML file

You shall use the **`engine`** field within the `variables` section.

### Syntax for Random Number Generation
In the `variables` block, the syntax typically follows this structure:

*   **engine:** Specifies the generator, which is usually **`numpy rng.`**.
*   **call:** Specifies the specific function to execute from that engine (e.g., `integers(0, 10, size=1)`).

### Example Implementation
An example from the documentation shows how to define two integer variables, `a` and `b`, with values between 0 and 10:

```yaml
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


### Key Details
*   **Variable Usage:** Once these are defined, you can use them in your question text or propositions by enclosing them in curly braces, such as `{a}` or `{b}`.
*   **Activation:** To ensure the quiz actually generates these random values when it is displayed in a notebook, you must set the **`autovars=True`** option in the `show()` function: `quiz.show("quiz_id", autovars=True)`.
*   **Alternative Methods:** You can also use the `quiz_editor` tool, which provides a graphical interface to define variable generation formulas like `rng.integers(1,10)` without manually editing the YAML code.
*   

###  Can I use the quiz_editor to define these variables visually?

Yes, you can use the **quiz_editor** to define variables visually, as it was specifically created to provide a graphical interface for editing and managing question files without needing to manually edit YAML code.

The `quiz_editor` includes several features that facilitate variable definition for template questions:

*   **Dedicated Variable Generation:** It allows you to explicitly **define variable generation for templates** within the interface.
*   **Formula Support:** You can enter generation formulas directly, such as `rng.integers(1,10)`, to create randomized values.
*   **Live Previews:** One of the most significant advantages of using the editor is the **preview feature**. You can see how automatically generated variables will appear in the question stem and propositions, including the formatting of numerical values.
*   **Visual Elements:** The editor can preview tables or scatter plots based on these generated variables, which is particularly useful for complex `numeric-template` or `MCQ-template` questions.
*   **Solution Calculation:** It allows you to use these variables to calculate expected solutions (Boolean or numerical) and verify them within the editor before exporting your quiz.

The application is available as a cloud-based tool (at `https://jfb-quizeditor.streamlit.app/`) or can be installed locally via `pip install quiz-editor`.

::::

::::{dropdown} How can I export quizzes to AMC-LaTeX for paper exams?

To export your quizzes to **AMC-LaTeX** for paper-based multiple-choice exams, you must use the **quiz_editor** tool, which includes a specific conversion feature for this purpose.

The export process utilizes the following mechanisms and options:

### 1. Mapping Content to AMC Structure
When converting your question file, the editor uses the **categories** defined in your YAML file to determine the **`\element` type** in the resulting LaTeX code. This allows you to organize your questions into specific groups within the AMC framework.

### 2. Export Options for Template Questions
The editor provides two distinct ways to handle randomized "template" questions:

*   **Static Export:** In this mode, the editor generates a single set of values for your variables based on your generation formulas (e.g., `rng.integers(1,10)`). These values are then **reused for every student copy**, meaning all students receive the same version of the randomized question.
*   **Dynamic Export:** This mode utilizes the **pythonTeX** LaTeX package to allow for **different values on every student's sheet**. This ensures that each exam copy is unique. This method requires:
    *   The installation of the `pythonTeX` package on your system.
    *   Adding **three specific lines** to the beginning of your LaTeX file.
    *   Using the provided **`prePythonTex4AMC`** utility.
    *   Consulting the templates and documentation available in the `extras/pythontex_template` directory of the suite.

### 3. Accessing the Tool
You can perform these exports using the cloud-based `quiz_editor` (at `https://jfb-quizeditor.streamlit.app/`) or by running it locally via `pip install quiz-editor` and invoking the `quiz_editor` command in your terminal. The interface includes an **export menu** where you can select the AMC-LaTeX format after selecting the questions you wish to include.

::::

::::{dropdown} How can I set up the Google Sheet for results?

To set up a Google Sheet for collecting results, you can choose between a quick template-based method or a manual configuration. This setup allows you to store student identifiers, answers, and the randomized variables used in template questions.

### 1. The Shortcut Method ("Simplest of the Simplest")
The easiest way to begin is by using the pre-configured template provided in the documentation.
*   **Copy the Template:** Open the provided template link and select **File > Create a copy**.
*   **Deploy the Script:** Navigate to **Extensions > Apps Script**. Click the **Deploy** button in the top right corner and select **New deployment**. 
*   **Configure Access:** Choose **Web Application** as the type and ensure you share it with **"Everyone"**. 
*   **Save the URL:** Copy and save the resulting link (ending in `/exec`). This is the **URL** you will use to link your quizzes and read collected data.

### 2. Manual Preparation
If you prefer to build the sheet from scratch, follow these steps:
*   **Step 1: Primary Sheet Headers:** Create a new Google Sheet. On the first line of the first sheet, insert the following headers: `timestamp`, `send_timestamp`, `notebook_id`, `student`, `quiz_title`, `event_type`, `parameters`, `answers`, and `score`.
*   **Step 2: Configuration Sheet:** Create a second sheet within the same file and rename it **"Config"**. 
    *   **A2 (Pwd):** Enter a key or password here. This key checks the connection and helps encrypt the question file.
    *   **B2 (Data Reception):** Insert a **Checkbox** via **Data > Data Validation**. **This box must be checked** for the sheet to receive data. 
    *   **Limits:** You can define `NMAX` and `NKEEP` values to limit the number of rows stored in the sheet.
*   **Step 3: Script Setup:** Go to **Extensions > Apps Script**. Delete any existing code and paste the content from the **`code_gs.txt`** file (found in the `extras` folder of the LabQuiz suite).
*   **Step 4: Set the Secret:** On the first line of the script code, replace the **`SECRET`** value with your chosen password. Ensure the `SHEET1` constant in the code matches your first sheet's name (e.g., "Feuille 1").
*   **Step 5: Deployment:** Click **Deploy > New deployment**, select **Web application**, and set access to **"Everyone"**. Copy the generated **URL**.

### 3. Usage and Security
Once established, you will need both the **URL** and the **SECRET** password to:
*   **Initialize the Quiz:** Pass these values into the `QuizLab` or `show()` functions in your notebook.
*   **Monitor Progress:** Use them in **`quiz_dash`** to track student submissions in real-time.
*   **Perform Correction:** Use them with the `correctQuizzes` function to retrieve and grade the recorded data.

**Note on Security:** For exams, enable **`mandatoryInternet=True`** to ensure all validations are recorded to the sheet immediately. You should also provide students with a version of the question file that does not contain the correct answers to prevent reverse-engineering.

::::

::::{dropdown} How to secure the LabQuiz YAML question file before deployment

To secure your LabQuiz YAML question file before deployment, especially for exams or graded assessments, you should follow these multi-layered strategies:

### 1. Remove Expected Answers (Primary Security)
The most critical recommendation for conducting an exam under good conditions is to use a question file **without the answers** (`expected` values). 
*   **Why:** Since Python source code is accessible within Jupyter notebooks, a motivated student can reverse-engineer the quiz files and potentially share the answers.
*   **Deployment strategy:** For high-stakes evaluations, distribute only the "answer-less" file to students, and use a version **containing the expected values** solely on the teacher's side for retrospective correction on the remote server.

### 2. Encryption and Encoding
The LabQuiz suite provides built-in mechanisms to protect your content from easy viewing:
*   **Quiz Encryption:** Files can be encrypted using a key calculated at runtime, which may depend on a key stored on a remote server.
*   **Quiz Editor:** The `quiz_editor` tool features a specific menu to **prepare an encoded or encrypted version** of your YAML file automatically.

### 3. Client-Side Integrity Checks
To prevent students from tampering with the quiz logic or the data sent to the server, you can implement an encrypted archive:
*   **Secure Tar Archive:** You can create an encrypted archive (e.g., `quiz.tar.enc`) containing a hidden module like `zutils.py`.
*   **Two-Password System:** This system uses a `password_open` (to decrypt and run the quiz) and a `password_seal` (to watermark sources).
*   **Integrity Daemon:** This archive can launch a background "daemon" that periodically performs integrity checks on the student's machine and transmits the status to the server.

### 4. Machine Identification and Hash Tracking
The suite automatically implements several background security measures:
*   **Hardware/Software ID:** Each machine is identified by a unique ID used in all transactions to detect if multiple students are sharing the same computer.
*   **Source Hashing:** A hash of the source code and critical parameters (like `exam_mode` or `retries`) is transmitted with every response. This allows teachers to use the `quiz_dash` or the `check_hash_integrity` function to detect if a student has modified the code or "monkey patched" the parameters.

### 5. Mandatory Internet Mode
By setting the `mandatoryInternet=True` parameter, you ensure that **all validations and correction requests are recorded** and transmitted in real-time. This prevents students from testing answers offline before submitting them.

::::

::::{dropdown} How do I set numerical tolerance for math questions?

Based on the documentation, you can set the numerical tolerance for math questions (defined as **`numeric`** or **`numeric-template`** types) by using two specific parameters within your question configuration:

*   **`tolerance`**: Used to define a **percentage-based margin of error** (for example, a value of `0.05` allows for a 5% variation from the correct answer).
*   **`tolerance_abs`**: Used to define a **fixed absolute value** as the accepted margin of error.

### Setting Tolerance via the Quiz Editor
In the **`quiz_editor`**, you can configure these settings by selecting the "numeric" question type from the interface. The editor allows you to edit the propositions and define the expected numerical solution. It also provides a preview feature so you can see how these parameters and calculated solutions will behave with automatically generated variables.

### Key Technical Requirements for Accuracy
To ensure the tolerance works correctly with the correction logic, keep the following rules in mind:

*   **Explicit Calculation:** For template questions involving random variables, you must compute the **`expected`** value explicitly using Python functions like **`round(value, precision)`**. 
*   **Avoid Relying on Display Formatting:** While you can use formatting in the question stem to change how a number looks to a student (e.g., `{a+b:3f}`), this is **only for display**. It does not affect the underlying correction logic, which still requires an explicit mathematical expression for the expected answer.
*   **Validation:** You can use the `quiz_editor` to calculate the expected solution in real-time and verify it before exporting your quiz file.
 

::::

::::{dropdown} What about security mechanisms to avoid fraud?

The LabQuiz suite implements a multi-layered security approach to prevent fraud, ranging from file-level encryption to real-time integrity monitoring.

### **1. File Encoding and Encryption**
The **quiz_editor** provides dedicated tools to prepare **encoded or encrypted versions** of your YAML question files. This is a critical first step to prevent students from simply opening the file to find the "expected" answers. For high-stakes exams, it is recommended to distribute a version of the question file that **does not contain the answers** at all, keeping the complete version only on the teacher's side for retrospective correction.

### **2. Randomization through Templates**
One of the most effective fraud prevention mechanisms is the use of **template questions**, which generate unique data for every student:
*   **Dynamic AMC-LaTeX Export:** For paper exams, this creates unique sheets for every student, making it impossible to copy numerical answers from a neighbor.
*   **Dynamic HTML:** In self-assessment web pages, variables are **dynamically regenerated** at startup and after each reset.
*   **HTML Exams:** These support real-time submission of results to a central database, ensuring students cannot test multiple answers offline before submitting.

### **3. Real-Time Integrity Monitoring**
When the quiz is running in a Jupyter notebook, several background mechanisms track the session's integrity:
*   **Machine Identification:** The system tracks a **unique machine ID** for every student. Using the `check_machine` function or the `quiz_dash` dashboard, instructors can instantly detect if multiple students are sharing the same terminal.
*   **Source Hashing:** The suite transmits a **hash of the quiz source code** and critical parameters (like the number of allowed retries or the active mode) to the server. This allows the **quiz_dash** tool to flag if a student has "monkey patched" the code to give themselves more attempts or change the scoring logic.
*   **Mandatory Internet Mode:** By enabling `mandatoryInternet=True`, you ensure that all student actions, validations, and integrity heartbeats are recorded on the remote server as they happen.

### **4. Access Control and Secure Archives**
*   **Google Authentication:** You can link quizzes to **Google Authentication**, which not only verifies student identities but also uses their credentials to encrypt the quiz file.
*   **Secure Tar Archives:** For advanced security, you can package the quiz in an **encrypted archive** alongside an integrity daemon that periodically checks the student's system status and reports anomalies to the teacher.

::::

::::{dropdown} Can a student modify quiz parameters?

Technically, **a student can modify quiz parameters** because the Python source code is accessible in a Jupyter notebook, but the LabQuiz suite is designed to **immediately detect and flag such modifications**.

### 1. Detection via Integrity Hashing
The system monitors a specific **"WATCHLIST"** of parameters, which includes **`exam_mode`**, **`test_mode`**, and **`retries`**.
*   **The "Big Hash":** A hash of these parameters (along with a hash of source code loaded in memory) is periodically transmitted to the results server. 
*   **Hash Mismatch:** If a student attempts to modify source or "monkey patch" the code the **calculated hash will change**.
*   **Teacher Alerts:** Tools like **`quiz_dash`** or the **`check_hash_integrity`** function will explicitly flag the student by name and machine ID, reporting exactly which key was changed and what its original value was.

### 2. Functional Failure
Modifying certain core parameters will cause the quiz to **stop functioning correctly**:
*   **Decryption Link:** If **`googleAuthentification=True`** is set, the quiz file is specifically encrypted to require that authentication. If a student modifies this parameter to `False` to try and bypass the login, they will be **unable to decrypt the quiz file** and cannot see the questions.
*   **Connectivity:** If **`mandatoryInternet=True`** is required, the quiz will raise an exception and refuse to run if it cannot verify a connection to the results server to record these parameters.

### 3. Continuous Monitoring (Daemon)
Even if a student modifies a parameter after the quiz has started, a background **daemon** periodically transmits the system status. This "heartbeat" ensures that the current state of the parameters in the student's memory is constantly compared against the original values expected by the teacher.

### 4. Advanced Security (Secure Archives)
For high-stakes exams, instructors can use the **`create_secure_tar`** tool to package the quiz in an **encrypted archive**. This includes an **autonomous integrity daemon** that is minified and obfuscated, making it significantly harder for a student to reverse-engineer or modify the monitoring process without being caught.

In summary, while the open nature of Jupyter allows students to edit the code, the **continuous transmission of state hashes** makes it "reasonably complex and difficult" to modify parameters successfully without alerting the instructor.

::::

::::{dropdown} Can a student modify sources to spoof the system?
Technically, **a student can modify the sources** because the Python code is accessible within the Jupyter notebook, but the LabQuiz suite is designed to **immediately detect and flag** such attempts through several layers of integrity monitoring.

### **1. Detection of "Monkey Patching"**
Since the environment is open, a student could try to "monkey patch" the code (modifying the Python objects or functions in memory) to change behavior, such as increasing the number of allowed attempts or disabling exam mode. The system counters this by:
*   **Source and Parameter Hashing:** The suite calculates a **"big hash"** that includes both the source code of the modules (like `main` and `utils`) and a **watchlist** of critical parameters: `exam_mode`, `test_mode`, and `retries`.
*   **Periodic Transmission:** This hash is periodically transmitted to the server via a background **daemon**. If a student changes even a single line of code or a monitored parameter, the hash will not match the "wanted hash" expected by the teacher, and the anomaly will be flagged on the **`quiz_dash`** dashboard.

### **2. Machine and System Identification**
Spoofing the system by switching computers or sharing answers is detected through **machine identification**. The suite captures a unique identifier based on the machine's hardware and software system, which is included in every transaction.
*   The `check_machine` function and the dashboard can identify if the **same machine ID** is being used by multiple student names, suggesting unauthorized collaboration.
*   It also detects **machine modification**, where a single student's record shows different machine IDs during the same session.

### **3. Encryption and Authentication Locks**
Some parameters are functionally locked through encryption:
*   **Google Authentication:** If `googleAuthentification=True` is enabled, the quiz file is specifically encrypted to enforce this. If a student tries to spoof the system by modifying this parameter to `False` to bypass login, they will be **unable to decrypt the quiz file** and cannot access the questions.
*   **Mandatory Internet:** When `mandatoryInternet=True` is set, the system forces all validations and heartbeats to be recorded in real-time. Bypassing this while still trying to submit answers is "reasonably complex and difficult".

### **4. Advanced Spoofing Protection (Secure Archives)**
For high-stakes exams, teachers can use the `create_secure_tar` tool to package the quiz in an **encrypted archive**. 
*   This archive contains an **autonomous version of the integrity daemon** (`zutils.py`) that has been **minified and obfuscated** (renaming globals, removing literals) to make reverse engineering significantly harder.
*   This daemon is watermarked with a `session_hash` and performs checks that are transmitted independently to the server.

### **Teacher Recommendation for High-Stakes Exams**
The documentation notes that a highly motivated student might still attempt to reverse engineer the data sent to the server. Therefore, for formal exams, the most effective protection is to **distribute a question file that does not contain the answers**. This ensures that even if a student successfully spoofs the monitoring system, they cannot find the correct answers within the local files.

::::

::::{dropdown} About Google authentication?

Regarding **Google authentication** in the LabQuiz suite, it is used as a security and identification layer that ensures only authorized students can access a quiz and that their results are correctly linked to their identity.

Drawing from the provided documentation for the `quiz_editor`, here is a breakdown of how it works and how it is configured:

### **1. Core Configuration**
To enable Google authentication for a quiz session, you must set specific parameters when initializing the quiz in your Jupyter notebook:
*   **Parameters:** You must set `needAuthentification=True`, `googleAuthentification=True`, and `mandatoryInternet=True`.
*   **Initialization Code:** The quiz is triggered by calling the authentication method:
    ```python
    if quiz.googleAuthentification:
        await quiz.googleAuthentify()
    ```.

### **2. Relationship with Security and Encryption**
Google authentication is deeply integrated with the suite's anti-fraud mechanisms:
*   **Decryption Key:** When authentication is enabled, the quiz file is encrypted in a way that requires successful authentication to be decrypted. If a student tries to bypass the login by changing the code, they will be unable to view the questions.
*   **Editor Preparation:** The **`quiz_editor`** includes a specific feature for the **"preparation of an encoded or encrypted version"** of question files. This ensures that even if students access the raw YAML file, they cannot see the answers without going through the authorized interface.

### **3. Infrastructure Setup (Google Cloud Console)**
Setting this up requires a one-time configuration in the Google Cloud Console to create credentials:
*   **OAuth Consent Screen:** Instructors must create a project and configure the consent screen, typically setting the user type to **"Internal"** to restrict access to students within their own institution.
*   **Credential Files:** Depending on the deployment, you must generate and rename specific JSON files and place them in the quiz directory:
    *   **`client_desktop.json`**: For quizzes running in local Jupyter notebooks.
    *   **`client_web.json`**: For quizzes deployed on the web via **JupyterLite**.

### **4. Result Collection**
Authentication is often paired with a **Google Sheet** backend. The **`quiz_editor`** allows for an **"HTML exam"** export that features real-time submission of results directly to a Google Sheet. This allows teachers to collect authenticated responses and use tools like **`quiz_dash`** to monitor progress and verify that the same student is not sharing their machine with others.

::::

::::{dropdown} How do I configure Google authentication for my results sheet?

To configure Google authentication for your results collection, you must set specific parameters in your quiz code and complete a one-time configuration in the **Google Cloud Console** to generate credential files.

### 1. Enable Authentication in the Quiz Code
When instantiating your quiz in a Jupyter notebook, you must include three mandatory parameters and a specific line of code to trigger the login process:

*   **Mandatory Parameters:**
    *   `needAuthentification=True`
    *   `googleAuthentification=True`
    *   `mandatoryInternet=True`
*   **Initialization Code:**
    ```python
    quiz = QuizLab(URL, QUIZFILE, needAuthentification=True, mandatoryInternet=True, googleAuthentification=True)
    if quiz.googleAuthentification:
        await quiz.googleAuthentify()
    ```

### 2. Google Cloud Configuration (One-Time Setup)
As an instructor, you must create a project at the [Google Cloud Console](https://console.cloud.google.com/) and follow these steps:

*   **Configure the OAuth Consent Screen:**
    1.  Go to **APIs & Services → OAuth consent screen**.
    2.  Select **Internal** as the User Type (this restricts access to members of your institution).
    3.  Fill in the required application name and support email.
*   **Create OAuth Credentials:**
    1.  Go to **APIs & Services → Credentials → Create Credentials → OAuth client ID**.
    2.  The setup depends on how students access the quiz:
        *   **For Local Notebooks:** Choose **Desktop app**. Download the resulting JSON file, rename it to **`client_desktop.json`**, and place it in the same directory as the student notebook.
        *   **For Web Applications (JupyterLite):** Choose **Web application**. You must add an **Authorized redirect URI** (the URL of your JupyterLite site). Download the JSON, rename it to **`client_web.json`**, and place it next to the notebook in the web application.

### 3. Security Considerations
*   **Enforced Authentication:** When `googleAuthentification=True` is enabled, the quiz file is specifically encrypted to require valid credentials. If students attempt to modify this parameter to bypass login, they will be unable to decrypt and view the quiz.
*   **Institutional Access:** By using the "Internal" user type, you ensure that only users with an institutional Google account can submit results to your sheet.
*   **IT Support:** If you find you cannot create a project or the "Internal" option is unavailable, you may need to contact your institutional IT administrator to grant the necessary permissions.


::::

::::{dropdown} What is the download link for the Jupyterlite archive?

The download link for the pre-deployed Jupyterlite archive is:

**[https://drive.google.com/file/d/14BnzVmPO6I8uOMEmNC6BTMsjXL3RLrQL/view?usp=sharing](https://drive.google.com/file/d/14BnzVmPO6I8uOMEmNC6BTMsjXL3RLrQL/view?usp=sharing)**

This archive is provided as a **"shortcut"** for teachers and developers because a full preparation and deployment of Jupyterlite can be complex and time-consuming. By downloading and unzipping this version, you can quickly integrate your own quizzes and notebooks by placing them in the `_output/files` directory and running the `update_jupyterlite_contents.py` script found in the `extras` folder.

::::


# quiz_editor


::::{dropdown} What is quiz_editor?

The **quiz_editor** is a graphical web application designed to simplify the creation, management, and export of LabQuiz question files, removing the need for manual YAML editing.

Its primary roles and features include:

### 1. Graphical Question Editing
*   **Rich Content Support:** It allows teachers to edit questions and propositions using **Markdown**, supporting formatting, lists, images, tables, and mathematical formulas.
*   **Comprehensive Fields:** The interface provides dedicated fields for editing question stems, options (correct/incorrect), **hints (tips)**, **explanations (answers)**, and scoring parameters like **bonuses and penalties**.
*   **Organization:** Teachers can define **categories and tags** to organize large databases of questions and filter them for specific uses.

### 2. Template and Variable Management
*   **Dynamic Generation:** The editor is used to define **variable generation for template questions** (e.g., using formulas like `rng.integers(1,10)`).
*   **Visual Previews:** It provides live previews of how these variables will appear in the question. This includes generating **dynamic tables** (like confusion matrices) or **scatter plots** (for linear regression) based on the randomized variables.
*   **Automatic Solutions:** It can automatically **calculate expected solutions** (boolean or numerical) based on the generated variable values, ensuring the quiz logic is correct before deployment.

### 3. Security and File Preparation
*   **Encryption and Encoding:** The editor features a menu to prepare **encoded or encrypted versions** of the question file to enhance security during exams.
*   **Selective Export:** Teachers can extract and save only a subset of questions from a larger file.

### 4. Format Conversion
*   **AMC-LaTeX:** It can convert questions into the **AMC-LaTeX** format for use in paper-based multiple-choice exams. It supports both static exports and dynamic exports (where each student's paper contains different randomized values).
*   **HTML Export:** It can generate **self-assessment web pages** (Dynamic HTML) with integrated answers or **web exams** (HTML exam) that submit results to a Google Sheet in real-time.

The tool is accessible as a **cloud application** (via Streamlit) or can be installed locally using `pip install quiz-editor`.

::::

::::{dropdown} How do I filter questions by categories and tags?

To filter questions by categories and tags, you primarily use the **quiz_editor**, which is designed to manage question databases and facilitate selective exports.

### 1. Defining Categories and Tags
Within the **quiz_editor** interface, you can assign specific **categories** and **tags** to each question. 
*   **Categories:** These are often used for broad classification and are particularly important when exporting to **AMC-LaTeX**, where they define the `\element` type for the exam structure.
*   **Tags:** These allow for more granular labeling and organization of your questions.

### 2. Filtering and Selection
The editor provides a dedicated mechanism to **select questions by filtering** on these defined labels. This allows you to:
*   Navigate large question databases more efficiently.
*   **Extract and save** only a specific subset of questions for a new YAML file.
*   Quickly identify questions for specific exports, such as creating a targeted practice test or a formal exam.

### 3. Visual Interface
The interface includes a **Markdown preview** and an editing panel where you can define these attributes. The filtering options are integrated into the question selection and export menus, allowing you to refine which questions are active for your current task.

While the **quiz_editor** is the main tool for filtering question content, the **quiz_dash** monitoring tool also offers filtering capabilities, but these are focused on **filtering results by student groups or classes** to analyze performance.

::::

::::{dropdown} How do I configure pythonTeX for dynamic exam generation?

To configure **pythonTeX** for dynamic exam generation—allowing each student's paper-based AMC exam to have different randomized values for template questions—you must follow a specific workflow and configuration process within the LabQuiz suite.

### 1. Prerequisite
You must have the **pythonTeX** package installed in your LaTeX environment. This package enables the execution of Python code directly within LaTeX to handle the randomization of variables during the exam compilation.

### 2. Export Process
In the **quiz_editor**, navigate to the export menu and select the **AMC-LaTeX** format. Choose the **"dynamic" export** option rather than the static one. This ensures that the Python formulas used for your variables are preserved in the LaTeX source code instead of being evaluated only once at the time of export.

### 3. Required Configuration Steps
According to the technical documentation and its footnotes, the following configuration is mandatory to make the dynamic export compatible with the Auto-Multiple-Choice (AMC) system:

*   **Add Configuration Lines:** You must add **three specific lines** to the very beginning of your LaTeX file.
*   **Utility File:** Use the **`prePythonTex4AMC`** file provided with the suite to bridge the gap between pythonTeX and the AMC software.
*   **Consult Templates:** A ready-to-use template and detailed examples are located in the **`extras/pythontex_template`** directory of the LabQuiz distribution.

### 4. How it Works
When configured correctly, the LaTeX file will use Python to regenerate variable values (like those defined with `rng.integers(1,10)`) for every individual student copy produced by AMC. This prevents students from sharing answers, as the numerical values and corresponding solutions in their confusion matrices or regression plots will differ from their neighbors'.

::::

::::{dropdown} What is the difference between static and dynamic HTML exports?

The **quiz_editor** offers two distinct HTML export options that differ in their purpose, the inclusion of answers, and how they handle dynamic content:

### 1. Dynamic HTML (Self-Assessment)
This export is designed for creating **"self-assessment" web pages**.
*   **Integrated Answers:** It includes both the answers and a correction mechanism, allowing users to check their own work immediately.
*   **Variable Regeneration:** It fully supports template questions. Values for these variables are **dynamically regenerated** at startup and after every reset.
*   **Technical Implementation:** This dynamic generation is handled by **JavaScript**, which interprets the initial Python formulas. It supports standard mathematical operations (such as `int`, `float`, `round`, `min`, `max`, `abs`), though it does not support complex Python-specific idioms or external packages.

### 2. HTML Exam (Static)
This export is intended for **formal assessments and exams**.
*   **No Answers:** Unlike the dynamic version, this export does **not include answers** or feedback to ensure the integrity of the test.
*   **Result Submission:** It features **real-time submission** of student results to a Google Sheet, which the instructor can then correct later or monitor via the `quiz_dash` dashboard.
*   **Current Limitations:** While support for dynamic variable generation (to make every student session unique) is in development, this export is currently **static**. This means that while it functions as an interactive web exam, it does not yet regenerate variables on the fly like the Dynamic HTML version.
   
::::

::::{dropdown} What encryption methods can the quiz_editor use?

The **quiz_editor** provides a dedicated interface and menu specifically for preparing **encoded or encrypted versions** of your question files to enhance security. 

Based on the documentation, the encryption methods and features include:

*   **Menu-Driven Encryption:** The editor features a specific menu to automate the process of preparing an encoded or encrypted version of a YAML question file, making it easier than manual configuration.
*   **Key-Based Encryption:** The suite supports encryption using **keys calculated at runtime** or keys that are stored on a **remote server** to prevent unauthorized viewing.
*   **Authentication-Linked Encryption:** When **Google Authentication** is enabled (`googleAuthentification=True`), the quiz file is specifically encrypted so that it can only be decrypted and viewed by students with valid institutional credentials.
*   **Secure Archive Preparation:** The editor facilitates the preparation of files for inclusion in **encrypted archives** (such as a `secure tar`), which can then be used alongside integrity checks to prevent tampering during exams.

These methods are designed to protect the "expected" answers in the YAML file from being reverse-engineered by students, which is considered a critical security practice for high-stakes evaluations.

::::

::::{dropdown} How do I filter questions by categories and tags?

To filter questions by categories and tags, you primarily use the **quiz_editor**, which is designed to manage question databases and facilitate selective exports.

### 1. Defining Categories and Tags
Within the **quiz_editor** interface, you can assign specific **categories** and **tags** to each question. 
*   **Categories:** These are often used for broad classification and are particularly important when exporting to **AMC-LaTeX**, where they define the `\element` type for the exam structure.
*   **Tags:** These allow for more granular labeling and organization of your questions.

### 2. Filtering and Selection
The editor provides a dedicated mechanism to **select questions by filtering** on these defined labels. This allows you to:
*   Navigate large question databases more efficiently.
*   **Extract and save** only a specific subset of questions for a new YAML file.
*   Quickly identify questions for specific exports, such as creating a targeted practice test or a formal exam.

### 3. Visual Interface
The interface includes a **Markdown preview** and an editing panel where you can define these attributes. The filtering options are integrated into the question selection and export menus, allowing you to refine which questions are active for your current task.

While the **quiz_editor** is the main tool for filtering question content, the **quiz_dash** monitoring tool also offers filtering capabilities, but these are focused on **filtering results by student groups or classes** to analyze performance.

::::

# quiz_dash


::::{dropdown} What is quiz_dash?

**quiz_dash** is a graphical monitoring and analysis dashboard designed for teachers to manage LabQuiz sessions in real-time and perform post-session evaluations. It provides a user-friendly interface to perform tasks that would otherwise require manual terminal commands.

Its roles can be categorized into four main areas:

### 1. Real-Time Monitoring and Tracking
*   **Live Submission Tracking:** Teachers can track participant submissions as they happen with an adjustable refresh rate.
*   **Progress Visualization:** It allows for viewing the progress of both individual students and the entire group over time.
*   **Participation Analytics:** The tool detects and quantifies early attempts, late attempts, and retakes occurring after the main session.

### 2. Integrity and Fraud Prevention
*   **Parameter Verification:** It checks that critical parameters—such as the number of allowed retries or the active mode (e.g., exam mode)—have not been tampered with by students.
*   **Hash Integrity:** The dashboard verifies the hash of sources, objects in memory, and their dependencies to ensure the quiz code has not been modified or "monkey patched".

### 3. Grading and Automated Correction
*   **Automated Scoring:** Teachers can automatically correct all recorded answers and retrieve a complete results table.
*   **Dynamic Adjustments:** It allows for the adjustment of the weight matrix (for multiple-choice questions) and the scoring scale for each individual question, with automatic score recalculation.
*   **Report Generation:** It can generate and download detailed grading reports in HTML or PDF format for individual students or the entire class.

### 4. Technical Flexibility and Accessibility
*   **Deployment:** It is available as a cloud application (via Streamlit) or can be installed locally using `pip install quiz-dash`.
*   **Data Persistence:** Since version 0.9, the dashboard is persistent, saving data in the browser's local storage so teachers can resume work if they close the browser or lose their connection.
*   **Group Management:** It supports filtering results by different classes or student groups.
  

::::

::::{dropdown} Where can I find the code_gs.txt file for the script?


The **code_gs.txt** file is located in the **extras folder** of the LabQuiz suite distribution.

This file contains the Google Apps Script code required for the manual preparation of a Google Sheet to collect quiz results. When setting up your script, you should:

1.  Open the **Apps Script** editor via the **Extensions** menu in your Google Sheet.
2.  Delete any existing code and paste the content from **code_gs.txt**.
3.  Replace the **`SECRET`** value on the first line of the code with your chosen password.
4.  Ensure the **`SHEET1`** constant in the code matches the name of your first sheet (e.g., "Feuille 1").

The `extras` directory also contains other useful tools, such as the `update_jupyterlite_contents.py` program used for deploying quizzes to Jupyterlite.

::::

::::{dropdown} How can I detect machine sharing during an exam?

To detect machine sharing during an exam, the LabQuiz suite uses a unique identifier assigned to each computer, which is tracked across all transactions recorded on the results server.

There are three primary ways to detect if multiple students are using the same machine:

### 1. Using the `check_machine` Function
The most direct method is to use the dedicated Python function provided in the `labquiz.putils` module. This function analyzes the filtered data from your results sheet and identifies instances where a single machine ID is associated with more than one student name.

*   **Implementation:** `from labquiz.putils import check_machine` followed by `check_machine(data_filt)`.
*   **Output Example:** The tool will explicitly flag the machine ID and list all associated student names, such as: `Same machine 04b1f154204fa9e9 used by several students ['Morane Bob', 'L Bobby', 'J Jean-Marc']`.

### 2. Monitoring via `quiz_dash`
The **`quiz_dash`** graphical dashboard provides a dedicated **Integrity Monitoring** tab that simplifies this process for instructors.
*   The dashboard automatically checks for inconsistencies in machine IDs and student names.
*   It allows you to visualize "machine modifications," which can indicate if a student has switched computers or if different students are logging into the same terminal.

### 3. Automated Background Checks (Daemon)
When a quiz is running, a background **daemon** periodically transmits the system status to the server, which includes the machine's software and hardware identification. This ensures that identification is not just captured during answer submissions but is monitored continuously throughout the session.

### Related Integrity Checks
In addition to machine sharing, the suite can detect other anomalies that may suggest fraud:
*   **Machine Modification:** Detecting if the same student name is linked to different machine IDs during a single session.
*   **Hash Integrity:** Verifying if a student has modified the source code, "monkey patched" parameters like the number of allowed retries, or altered the `exam_mode` setting.
*   **Mandatory Internet Mode:** By enforcing `mandatoryInternet=True`, you ensure that all these identification and integrity heartbeats are successfully recorded on the remote server.
  

::::

::::{dropdown} Tell me more about the CHECKALIVE integrity heartbeat.

The **`CHECKALIVE`** integrity heartbeat is a background mechanism that periodically transmits system status and identification data to the results server to ensure the security and continuity of a quiz session.

### **1. Function and Timing**
*   **Purpose:** It functions as a "heartbeat" to verify that the student’s environment remains intact and that no unauthorized modifications have been made during the session.
*   **Default Interval:** The heartbeat is controlled by the **`CHECKALIVE`** parameter in the `QuizLab` constructor, which defaults to **600 seconds** (10 minutes).
*   **Asynchronous Daemon:** The check is performed by an asynchronous loop (`check_alive`) that runs in the background without blocking the student's work in the notebook.

### **2. Data Transmitted**
During each heartbeat, the following information is recorded in the results sheet under the `event_type="check_integrity"` and `quiz_id="integrity"`:
*   **Machine Identification:** The unique software and hardware ID of the computer is sent to detect if a student has switched machines or if multiple students are sharing a terminal.
*   **Integrity Hash:** The system calculates a **"big_hash"** of the sources and a specific **WATCHLIST** of parameters (including `exam_mode`, `test_mode`, and `retries`). This allows the instructor to detect if a student has performed "monkey patching" to bypass exam rules.
*   **System Status:** It tracks whether the session is still active or if an inactivity timeout has been reached.

### **3. Operational Management**
*   **Stopping the Heartbeat:** The heartbeat only transmits if the quiz is currently active and the `keep_alive` flag is set. It can also be effectively stopped by **unchecking the "Data reception" box** in the `Config` tab of your Google Sheet. 
*   **Sheet Storage Impact:** Because this check is sent periodically, it can fill up the Google Sheet if left running for a long time. The sheet handles this by having a defined maximum number of rows (`NMAX`) and a cleanup threshold (`NKEEP`).
*   **Secure Archive Version:** When using a **secure encrypted archive** (`secure tar`), an autonomous version of this daemon (contained in the `zutils.py` module) is launched to perform even more rigorous integrity checks on the student's local configuration.

### **4. Monitoring via `quiz_dash`**
Instructors can monitor these heartbeats in real-time using the **`quiz_dash`** dashboard. The dashboard interprets the received hashes and machine IDs to flag anomalies, such as:
*   **Machine modification** for the same student name.
*   Changes to original keys (e.g., changing `exam_mode` from `True` to `False`).
*   Modification of the number of allowed `retries`.
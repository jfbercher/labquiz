## 🔐 Google Authentication in QuizLab (Notebook / Web)

For instructors whose institution uses Google Workspace, it is possible, since version 1.0.0, to use Google authentication instead of the standard authentication enabled with `needAuthentification=True`.
In that case, the parameter `googleAuthentification=True` must also be specified, in addition to the standard `mandatoryInternet=True`.

This short guide explains how to configure Google authentication and how to use it in QuizLab.

**Note:** The quiz file is specifically encrypted to enforce authentication when `googleAuthentification=True` is enabled. If students modify this parameter, they will not be able to decrypt the quiz file.

---

### Enable authentication in QuizLab

In your code, use something like

```python
quiz = QuizLab(
    URL,
    QUIZFILE,
    needAuthentification=True,
    mandatoryInternet=True,
    googleAuthentification=True,
    retries=100,
    groups=['A', 'B', 'C']
)

if quiz.googleAuthentification:
    await quiz.googleAuthentify()
```

👉 The following parameters are **mandatory** for authentification:

* `needAuthentification=True`
* `mandatoryInternet=True`
* `googleAuthentification=True`

---

###  For instructors: Google Cloud configuration (once)

Go to: https://console.cloud.google.com/
(Create a project if needed)

In Google Cloud:

#### a. OAuth consent screen

* Menu: **APIs & Services → OAuth consent screen**
* User type: **Internal**
* Fill in app name and support email

---

#### b. Create OAuth credentials

Menu: **APIs & Services → Credentials → Create Credentials → OAuth client ID**

---

🖥️ **Case 1 — Notebook / Local Python**

If you intend your students to use a local desktop version of jupyter notebook or jupyter lab

* Application type: **Desktop app**
* Download the JSON file
* Rename it:

```bash
client_desktop.json
```

* Place it in the same directory as the notebook

---

🌐 **Case 2 — Web application (e.g. JupyterLite)**

* Application type: **Web application**

* Add an **Authorized redirect URI**:

  * e.g. `http://localhost:8000/` or your site URL e.g. `https://www.my_university.edu/~mypage/my_jupyterlite_page.html`

* Download the JSON file

* Rename it:

```bash
client_web.json
```

* Place it next to the notebook in the web app

---

### IT support (if required)

Contact your IT administrator if:

* you cannot create a project
* “Internal” option is unavailable
* OAuth access is restricted

---

### ✅ Result

* Users authenticate with their institutional Google account
* Access is restricted to the organization

---

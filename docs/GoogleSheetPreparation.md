---
title: Google Sheet preparation
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

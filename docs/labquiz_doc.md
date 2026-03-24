---
title: LabQuiz 
subtitle: Une suite d'outils pour intégrer des quizzes dans des notebook Jupyter
#doi: 10.14288/1.0362383
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
    enabled: true
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

```{note} LabQuiz en quelques mots
 
 C'est un package python qui
 - permet d'intégrer des questions sous forme de QCM ou de valeurs numériques,
 - avec tentatives multiples (ajustable), indices et *feedback*, 
 - de corriger les quizzes en ligne ou *a posteriori*, 
 - d'authentifier et enregistrer automatiquement les réponses sur un serveur distant,
 - de disposer d'un *dashboard* pour suivre la progression des étudiants,
 - et plus encore
```

```{tip} Objectifs pédagogiques

- favoriser l'implication des élèves par des exercices et une évaluation *ludique*,
- contrôler et mesurer les résultats à obtenir pendant la séance,
- permettre de *prendre du recul* en reliant les résultats au cours et en approfondissant,
- favoriser l'apprentissage par des indicateurs de progression et des *feedbacks*,
- suivre, à l'aide d'un *dashboard*, la progression de l'étudiant au cours du temps,
- ainsi que du groupe au complet,
````


```{seealso} Liens

**Code source**

- https://github.com/jfbercher/labquiz
- https://github.com/jfbercher/quiz_editor
- https://github.com/jfbercher/quiz_dash

**En ligne**

- https://jfb-quizeditor.streamlit.app/
- https://jfb-quizdash.streamlit.app/

``` 

```{hint} Table des matières
Pour afficher une table des matières, presser le bouton table des matières sur la gauche ; ou menu `View/Table of Contents`
```

# Premiers exemples

Les *quizzes* sont intégrés dans le sujet tout au long de l'exercice de travaux dirigés ou pratiques, à partir d'un questionnaire préparé au préalable. 

Un premier exemple pour illustrer un QCM, avec tips et correction : 

:::{figure} doc_images/quiz2.gif
:label: fig1
:name: quiz2
:alt: Exemple de quiz
:align: center
:width: 60%

Question lors du TP
:::


Second exemple avec l'entrée de valeurs numériques

:::{figure} doc_images/quiz59.gif
:name: quiz2
:label:fig2
:alt: Exemple de quiz
:align: center
:width: 60%

Autre question lors du TP
:::


 (avec bloquage du quiz si la correction a déjà été demandée ! 😊)

```{warning} Mode connecté

**En mode connecté**, c'est-à-dire dès qu'on a passé une URL valide à l'initialisation

- toutes les entrées (Valider, Corriger) sont enregistrées et transmises,
- un identifiant spécifique de chaque machine est calculé et utilisé pour identifier tous les logs,
- un état du système est transmis périodiquement, avec une période connue et vérifiable, permettant de détecter 
    - des modifications des paramètres (mode examen, nombre d'essais autorisés, etc),
    - des modifications du source du package,
    - l'injection de code, 
    - un redémarrage du noyau, une nouvelle instanciation, un changement de machine...
    - etc

Bien entendu, une interruption de la transmission périodique par une machine dûment enregistrée est forcément détectée...
````


Dans ce qui suit
- le label 🧑‍🏫🏫 désigne des informations spécifiques aux enseignants ou concepteurs de sujet
- le label 🧑‍🎓 désigne des informations spécifiques aux étudiant.es

(installation)=
# Installation

🆙 Pour le moment, et avant une distribution éventuelle sur pypi, pour installer LabQuiz sur votre sytème, consulter
[ce dépôt](https://perso.esiee.fr/~bercherj/JliteNotes/pypi/), retenir le numéro de version x.y.z le plus élevé et 
saisir
```{code}
pip install https://perso.esiee.fr/~bercherj/JliteNotes/pypi/labquiz-x.y.z-py3-none-any.whl  --force-reinstall
```
dans un terminal, ou `%pip ...` dans une cellule de notebook. 

Il est possible d'utiliser une version autonome, *sans la moindre installation python*, qui tourne dans le navigateur. Ceci est décrit dans la [](jupyterlite).

(utilisation)=
# Utilisation

Une fois labquiz installé, on l'importe par
```{code}
import labquiz
from labquiz import QuizLab
```
et on instancie un quiz par
```{code}
URL = "" # chemin vers une URL pour recueillir les résultats
QUIZFILE = "nom_du_fichier_de_quiz" # par exemple "quizzes_basic_filtering_test.yaml"
quiz = QuizLab(URL, QUIZFILE)
```
Des paramètres supplémentaires peuvent être précisés (valeurs par défaut ci-dessous)
```
needAuthentification=True,  # Authentification nécessaire oui/non
retries=2,                  # nombre de tentatives possibles = retries + 1
mandatoryInternet=False,    # obligation d'avoir une connexion internet valide
CHECKALIVE=60,              # contrôle d'intégrité toutes les CHECKALIVE secondes
                            #(pas de modif du programme et des paramètres)
```                            

:::{figure} doc_images/login.png
:name: login
:label:fig
:alt: Login
:align: left
:width: 60%

Exemple de login
:::

# Fonctionnalités

⚠️ LabQuiz fonctionne aussi dans **visual studio code**, mais les portions de $\LaTeX$ ne sont pas rendues dans les questions, propositions et réponses. C'est une limitation de visual studio, qui ne charge pas MathJax, et cela sera peut être amélioré un jour. En attendant vaut mieux utiliser un jupyter classique, colab, ou un JupyterLite. Dans visual studio, cela peut se gérer, mais c'est moins bien s'il y a du $\LaTeX$ dans les textes. 

## Types de questions de quiz
Quatre types de questions sont disponibles : 
- des questions de qcm (`type: "qcm"` dans le fichier de questions ; c'est le type par défaut)
- des questions numériques (`type: "numeric"`)
- des question de qcm dépendant du contexte (`type: "qcm-template"`)
- des questions numériques dépendant du contexte (`type: "numeric-template"`)

La structure du fichier de question, où sont indiqués ces différents types, est présentée [](structure_fichier_question). La manière de préparer, voire de crypter ce fichier est présentée en [](prepa_encode_file). 

Les `templates` permettent de poser des questions dépendant de variables locales. On peut ainsi tester des valeurs précises, des ordres de grandeur, le résultat d'un calcul, une cohérence entre plusieurs valeurs. Deux exemples pour illustrer les choses :

:::{figure} doc_images/quiz5354.gif
:name: quiz3
:label:fig3
:alt: Exemple de quiz
:align: center
:width: 60%

Questions `template` qui utilisent des valeurs numériques passées en paramètre
:::


## Différents modes de présentation

### Mode apprentissage
En mode apprentissage, les 4 boutons sont présents. Le bouton valider affiche le score obtenu, cf [](fig5). Le bouton tips affiche des conseils. Le bouton corriger quant-à-lui affiche la correction, voir [](fig6) et [](fig7). Les cases sont passées en vert si elles ont été cochées ou non cochées à bon escient, et en rouge sinon. Les coches entrées par l'utisateur sont conservées, mais colorées en vert ou rouge suivant le résultat correct. Après que la correction ait été demandée, les boutons valider, reset et tips sont invalidés et deviennet inopérants. 

:::{figure} doc_images/4boutons_submit_actif.png
:name: quiz3
:label:fig5
:alt: Exemple de quiz
:align: left
:width: 60%

Mode apprentissage - le bouton validé à été pressé
:::

:::{figure} doc_images/4boutons_correction_actif.png
:name: quiz3
:label:fig6
:alt: Exemple de quiz
:align: left
:width: 60%

Mode apprentissage - le bouton corrigé à été pressé
:::


:::{figure} doc_images/exemple_correction.png
:name: quiz3
:label:fig7
:alt: Exemple de quiz
:align: left
:width: 60%

Mode apprentissage - le bouton corrigé à été pressé et la correction présentée
:::

### Mode test

Dans le mode test, le bouton corriger est supprimé. L'étudiant voit son score après validation et les tips son possibles. Le nombre de validation est limité par le paramètre `retries`passé à l'initialisation.

:::{figure} doc_images/3boutons_submit_actif.png
:name: quiz3
:label:fig8
:alt: Exemple de quiz
:align: left
:width: 60%

Mode test - le bouton validé à été pressé
:::

### Mode examen

En mode examen, il n'y a ni affichage du score, ni tips ni correction. Seuls apparaissent les boutons reset et valider. 

:::{figure} doc_images/2boutons_submit_actif.png
:name: quiz3
:label:fig9
:alt: Exemple de quiz
:align: left
:width: 40%

Mode examen - le bouton validé à été pressé
:::

### Questions individuelles ou série de questions `exam_show`

Indépendamment des modes présentatioin décrits ci-dessus, les questions peuvent être présentées soit individuellement avec un appel de la forme `quiz.show('label')`, soit en un bloc de questions. Cette dernière option peut être utile pour faire un petit bilan intermédiaire par exemple. 
Un bloc de questions sera présenté en utilisant la fonction `quiz.exam_show()` avec les paramètres suivants :
```
exam_show(exam_title="", questions=None, shuffle=False, nb=None)
```
- exam_title : Utilisé pour identifier l'examen, efault ""
- questions :  Liste de labels de questions (ex. "quiz1", "quiz2").
             Si None, utilise toutes les questions de la banque.
- shuffle : default False ; Si True, mélange l’ordre des questions avant l’affichage.
- nb : Si différent de None, tire aléatoirement nb questions distinctes
            parmi l’ensemble des questions disponibles.
            
Un exemple d'appel pourrait être 
```
ql = quiz.exam_show(exam_title="Test pour voir", shuffle=True, nb=4)
```
Les résultats intermédiaires ne sont pas affichés (si mode apprentissage ou test). 
Les résultats obtenus sont ensuite consultables par l'étudiant avec `quiz.exam_result(ql, bareme=None)`. 
En mode examen, les résultats ne sont pas calculés ni consultables, et l'enseignant pourra corriger l'ensemble des examens comme décrit dans [](correction_par_enseignant). 

(structure_fichier_question)= 
## 🧑‍🏫🏫 - Structure du fichier de questions

### Structure générale
Le fichier de questions est un fichier [YAML](https://en.wikipedia.org/wiki/YAML). Le format est simple, mais le nombre d'espaces ou d'indentation doit être cohérent tout au long du fichier. 

Le fichier débute par une ligne
```
title: une explication du contenu du fichier
```
Il comporte ensuite une liste de quizzes, débutant chacun par un label, par exemple (mais pas de contrainte sur le choix des labels
```
quiz1:
    ...
quiz2:
    ...
    
```
Chaque quiz en lui même comporte une question, puis une suite de propositions. 
- Le nombre de propositions n'est pas limité ; il en faut au moins une.  
- Les textes des questions et propositions sont des chaînes de caractères, qui peuvent être entourées de guillemets simples ou doubles. Ces guillemets ne sont pas obligatoires, sauf si la chaîne contient un :, auquel cas vous l'entourerez avec des guillemets simples (et dans ce cas, si la chaîne contient aussi des quotes simples, il faut les doubler, cf les exemples ci-dessous)
```
quiz23:
    question: Ceci est le texte de la question
      ...
    propositions:
        - proposition: texte de la première proposition
          ...
        - proposition: 'texte de la seconde proposition avec un : qu''il faut prendre en compte'
          ...    
```
C'est le strict minimum. 
Par défaut, le quiz est de type "qcm". Il peut aussi être de type "numeric" et dans ce cas il faut le préciser. Si vous voulez pouvoir utiliser la correction en ligne, présenter des tips, il va falloir les ajouter. 

(structure_type_qcm)= 
### Type QCM

```
quiz23:
    question: Ceci est le texte de la question
    type: "qcm"         #"qcm" ou "numeric" (optionnel - "qcm" par défaut)
    propositions:
        - proposition: texte de la première proposition (fausse)
          type : bool        # optionnel - par défaut "bool" si type "qcm", "float" si type "numeric"
          label: label1      # optionnel mais nécessaire pour les corrections
          expected: false    # valeur attendue pour la réponse
          tip: texte d'un indice ou d'une orientation vers la bonne réponse
          reponse: texte explicatif pour la bonne réponse
        - proposition: 'texte de la seconde proposition (vraie) avec un : qu''il faut prendre en compte'
          label: label2
          type : bool
          expected: true
          tip: texte d'un indice ou d'une orientation vers la bonne réponse, avec quotes '' si nécessaire
          reponse: texte explicatif pour la bonne réponse, avec quotes '' si nécessaire
```
- Des *contraintes de cohérence* sur les propositions peuvent être ajoutées. Par exemple, on peut imposer que la réponse vrai à la proposition de label `label2` implique que la réponse à la proposition `label1`soit fausse. En cas de violation, un malus est appliqué.
- De même, certaines propositions peuvent donner lieu à un *bonus* ou à un *malus*. Le bonus est le nombre de points accordés si la réponse est celle attendue (par défaut 1) et le malus est le nombre de points retirés si la réponse est différente de celle attendue (par défaut 0). 

Avec ces élements, l'exemple pourrait être complété comme ci-dessous. La mise en oeuvre est donnée ensuite [](fig10). 

```
quiz23:
    question: Ceci est le texte de la question
    type: "qcm"         #"qcm" ou "numeric" (optionnel - "qcm" par défaut)
    constraints: [
          { "indices": ["label2", "label1"], "type": "IMPLYFALSE", "malus": 2 }
          ] 
    propositions:
        - proposition: texte de la première proposition (fausse)
          type : bool        # optionnel - par défaut "bool" si type "qcm", "float" si type "numeric"
          label: label1      # optionnel mais nécessaire pour les corrections
          expected: false    # valeur attendue pour la réponse
          tip: texte d'un indice ou d'une orientation vers la bonne réponse
          reponse: texte explicatif pour la bonne réponse
        - proposition: 'texte de la seconde proposition (vraie) avec un : qu''il faut prendre en compte'
          label: label2
          type : bool
          expected: true
          tip: texte d'un indice ou d'une orientation vers la bonne réponse, avec quotes '' si nécessaire
          reponse: texte explicatif pour la bonne réponse, avec quotes '' si nécessaire
        - proposition: texte d'une troisième proposition avec malus
          label: label3
          type : bool
          expected: true
          malus: 2     #malus appliqué ici si la case n'est pas cochée
          tip: texte d'un indice 
```



:::{figure} doc_images/quiz23.gif
:name: quiz23
:label:fig10
:alt: Exemple de quiz
:align: center
:width: 60%

Mise en oeuvre de la question `quiz23`. On notera au passage que les propositions sont automatiquement mélangées. 
:::

Plusieurs contraintes logiques peuvent être spécifiées, comme dans cet exemple : 
```
quiz57:
  question: "Ceci est une question avec des contradictions et implications. Le nombre est 6"
  type: "qcm"
  constraints: [
      { "indices": ["parité", "impair"], "type": "XOR", "malus": 2 },
      { "indices": ["parité", "multiple 2"], "type": "SAME", "malus": 2 },
      { "indices": ["parité", "plus1pair"], "type": "XOR", "malus": 2 },
      { "indices": ["parité", "plus1impair"], "type": "IMPLY", "malus": 2 }
    ]
``` 


### Type `numeric`

Pour les questions à valeurs numérique, le schéma est similaire. On dispose des clés supplémentaires `tolerance`, `tolerance_abs`
- `tolerance` est le pourcentage de variation toléré sur la valeur attendue
- `tolerance_abs` est la toléance absolue. 
La tolérance retenue lors de la correction est la plus grande valeur des valeurs entre tolerance_abs et tolerance*attendue. 
Le `type` dans chaque proposition peut être `float`(par défaut) ou `int`. 
Des bonus (par défaut 1) et malus (par défaut 0) peuvent également être précisés et sont appliqués selon que la différence entre la valeur donnée et celle attendue sont supérieure ou inférieure à la tolérance. 

```
quiz24:
  question: Veuillez entrer ci-dessous le nombre de points et les valeurs de la moyenne et de l'écart type de la série temporelle
  type: numeric
  propositions:
    - proposition: Moyenne : 
      label: moyenne
      type: float
      expected: 0.0
      reponse: 0
      tolerance: 0.05
      tolerance_abs: 0.01
      tip: Entrer la valeur
    - proposition: Écart type
      label: sigma
      type: float
      expected: 1.0
      reponse: "1"
      tolerance: 0.05
      tolerance_abs: 0.01
      tip: Entrer la valeur
    - proposition: Nombre de points
      label: N
      type: int
      expected: 512
      reponse: Le nombre de points `len(serie)` ou `serie.shape[0]` est de 512
      tolerance: 0.01
      tolerance_abs: 2
      bonus: 2
      malus: 3
      tip: Entrer la valeur
```

:::{figure} doc_images/quiz24.gif
:name: quiz24
:label:fig11
:alt: Exemple de quiz
:align: center
:width: 60%

Mise en oeuvre de la question `quiz24` de type numérique. On notera que les propositions sont automatiquement mélangées et qu'on applique bien un bonus/malus sur le nombre de points. 
:::

### Cas des `templates`

Deux types supplémentaires sont possibles, à savoir les `numeric-template` et `qcm-template`. Ces formats permettent d'utiliser des données et valeurs numériques variables liées aux expérimentations menées dans le TP. Ces données dont passées en paramètre à la fonction `show()` et utilisées pour la correction. Pour des QCM, on pourra tester l'appartenance du résultat à un intervalle ou d'autres conditions calculables dont le résultat est booléen. Pour la correction, on implante une formule qui calcule la valeur attendue en fonction des paramètres passés. 
Dans l'exemple suivant, la fonction `show` est appelées avec deux paramètres :
```
quiz.show("quiz54", a=res1, b=res2)
```
Ces deux paramètres sont utilisés pour le calcul de la solution. Par exemple la formule `f'{a+b:.4f}'` (invisible à l'étudiant !) est *évaluée* avec le contexte qui est 
passé à la fonction, et qui en parallèle est *sauvegardé* sur le serveur distant afin de permettre à l'enseignant de recalculer la solution en différé. 

```
quiz54:
  question: "Ceci est une question numérique où il faut faire la somme et la différence de {a} et {b}"
  type: "numeric-template"
  propositions:
    - proposition: "Somme : "
      label: "somme"
      type: "float"
      expected: f'{a+b:.4f}'
      reponse: f'{a+b:.4f}'
      tolerance: 0.01
      tolerance_abs: 0.01
      tip: "Entrer la valeur"
    - proposition: "Différence : "
      label: "difference"
      type: "float"
      expected: f'{a-b:.4f}'
      reponse: f'{a-b:.4f}'
      tolerance: 0.01
      tolerance_abs: 0.01
      tip: "Entrer la valeur"
```

:::{figure} doc_images/quiz54.png
:name: quiz54
:label:fig12
:alt: Exemple de quiz
:align: center
:width: 60%

Mise en oeuvre de la question `quiz54` de type numérique. Des paramètres sont passé à la fonction, qui évalue et calcule la solution correcte à partir de ces paramètres. 
:::

À peu près n'importe quelle fonction python peut être utilisée pour l'évaluation de la réponse. 
Il faut simplement penser au fait que le contexte est sauvegardé dans le tableur distant. Il faut donc éviter que celui-ci soit trop volumineux : éviter les gros tableaux de données ! et préférer des contextes de taille réduite. Tous les contextes et types de données ne sont pas forcément sérialisables (dictionnaires, listes, tableaux numpy, pandas sont supportés ici). Par ailleurs, pour recalculer la solution en différé, il faut stocker quels sont les modules utiles. 

Dans l'exemple suivant, on calcule le coefficient de variation d'une série à l'aide de numpy. L'appel serait par exemple
```
quiz.show("quiz61", s=np.random.randn(5), np=np)
```
où l'on passe le nom du ou des modules utilisés dans le contexte. 

```
quiz61:
  question: "Ceci est une question numérique où il faut faire le calcul du coefficient de variation"
  type: "numeric-template"
  propositions:
    - proposition: "Coef de variation"
      label: "cv"
      type: "float"
      expected: f'{np.std(s)/np.mean(s)}'
```

:::{figure} doc_images/quiz61.png
:name: quiz61
:label:fig13
:alt: Exemple de quiz
:align: center
:width: 60%

Mise en oeuvre de la question `quiz61` de type numérique. Des paramètres sont passé à la fonction, dont un module, qui évalue et calcule la solution correcte à partir de ces paramètres. 
:::



```python

```

(calcul_du_score_correction)= 
## Calcul du score et correction par l'enseignant

(calcul_du_score)= 
### Calcul du score

Un score est calculé automatiquement pour chaque quiz, et affiché dans les modes apprentissage et test. Un score global, moyenne des résultats sur les quizzes effectués, est accessible par `quiz.score_global`. Le calcul du score repose sur les deux fonctions `calculate_quiz_score` et `correct_ans` disponibles respectivement dans les modules `utils` et `putils`. Les grands principes sont les suivants :
- La fonction reçoit la réponse de l'utilisateur, une matrice de poids, les contraintes imposées
- **matrice de poids**  - la *matrice de poids* sert à pondérer les réponses données en fonction de la réponse attendue : nombre de points accordés si l'utilisateur a répondu vrai alors que la réponse attendue était vraie, s'il a coché vrai alors que la réponse attendue était fausse, etc. La matrice par défaut est la suivante : 
```
        default_weights = {
                (True, True):   1,  # Vrai Positif  # A coché et la bonne réponse était vraie
                (True, False): -1,  # Faux Positif  # A coché alors que bonne réponse était fausse
                (False, True):  0,  # Faux Négatif  # N'a pas coché alors que la bonne réponse était vraie
                (False, False): 0   # Vrai Négatif  # N'a pas coché et la bonne réponse était effectivement fausse
            }
```
En clair on accorde ici un point pour les bonnes réponses cochées et on retire un point pour les bonnes réponses non cochées. Selon les cas on pourrait vouloir adapter ceci pour pénaliser les bonnes réponses non cochées, ou donner moins de poids à un oubli, par exemple
```
        weights = {
                (True, True):   1,   # Vrai Positif    # A coché et la bonne réponse était vraie
                (True, False): -0.5, # Faux Positif    # A coché alors que bonne réponse était fausse
                (False, True): -0.5, # Faux Négatif    # N'a pas coché alors que la bonne réponse était vraie
                (False, False): 0    # Vrai Négatif    # N'a pas coché et la bonne réponse était effectivement fausse
            }
```

On peut aussi imposer un poids *identité*, ce qui signifie que l'on accorde 1 point par proposition vrai cochée et 1 point par proposition fausse non cochée. Un inconvénient de cette stratégie est qu'évidemment une copie vierge sans aucune coche obtient statistiquement la moitié des points, si autant de mauvaises propositions que de bonnes, voire bien plus si on est dans un questionnaire ``une seule bonne réponse exacte''. 
```
        weights = {
                (True, True):   1,   # Vrai Positif    # A coché et la bonne réponse était vraie
                (True, False): 0, # Faux Positif    # A coché alors que bonne réponse était fausse
                (False, True): 0, # Faux Négatif    # N'a pas coché alors que la bonne réponse était vraie
                (False, False): 1    # Vrai Négatif    # N'a pas coché et la bonne réponse était effectivement fausse
            }
```


La matrice de poids n'est pas modifiable dans le cas du calcul du score en ligne, par contre elle peut être modifiée dans le cas du recalcul a posteriori à partir des résultats enregistrés ; voir la correction par l'enseignant [](correction_par_enseignant)

- **bonus malus** - Comme on l'a vu dans la structure du fichier [](structure_fichier_question), des `bonus` `malus` peuvent être intégrés dans les propositions elle même. ceci permet, pour une question donnée, de donner plus de poids à une bonne réponse particulière, ou on contraire de pénaliser une mauvaise réponse, et ceci indépendamment de la matrice de poids générale. 

- **contraintes logiques** - Des *contraintes logiques* peuvent être intégrées aux questions et utilisées pour le calcul du score. Ces contraintes sont spécifiées question par question au niveau du fichier de questions, et des malus appliqués en cas de violation de la contrainte. Les contraintes utilisables sont les suivantes : 
```
    # constraints: Liste de dicts ex: [{"indices": (0, 1), "type": "XOR", "malus": 2}]
    - XOR (Exclusion) A et B doivent être différentes
    - IMPLY (Implication) Si A est VRAI alors B est VRAI
    - SAME (Cohérence) A et B sont équivalentes (même valeur)
    - IMPLYFALSE: Si A est VRAI, alors B DOIT être FAUSSE
```
Voir un exemple à la fin de [](structure_type_qcm). 

- **valeurs numériques** - Dans le cas de réponses à valeurs numériques, la différence entre la valeur donnée et la valeur attendue est calculée. Si cette diférence est inférieure au seuil défini par la tolérance, la réponse est comptée juste[^1].
Rappelons que la tolérance est précisée au niveau du fichier de questions, et que l'on retient la plus grande valeur des valeurs entre tolerance_abs et tolerance_relative*attendue. Si la tolérance relative `tolerance`n'a pas été précisée, la valeur utilisée est de 1%. 
Des bonus (par défaut 1) et malus (par défaut 0) peuvent également être  appliqués selon que la différence entre la valeur donnée et celle attendue sont supérieure ou inférieure à la tolérance. 
[^1]: Il serait possible de fixer le score en fonction de la valeur (relative) de cette différence, mais cela n'a pas été fait et on verra plus tard.  


(correction_par_enseignant)= 
### Correction par l'enseignant

En parallèle du déroulé du TP, ou *a posteriori*, l'enseignant peut utiliser les résultats enregistrés automatiquement pour surveiller la progression, adapter les poids (matrice de poids) ou le barème. Évidemment, c'est indispensable en `mode examen` où les étudiants n'ont pas de feedback sur leurs réponses et où la correction n'est pas accessible. 

Pour corriger l'ensemble des réponses enregistrées, la correction est aussi simple que :
```python
from labquiz.putils import correctQuizzes

URL = "https://URL_UTILISÉE_POUR_RECUEILLIR_LES_RÉSULTATS"
SECRET = "MOT_DE_PASSE_SECRET_SPÉCIFIÉ_DANS_LE_SHEET"
QUIZFILE = "NOM_DU_FICHIER_DE_QUESTIONS.yml" #fichier de quiz CONTENANT les valeurs attendues
#
Res = correctQuizzes(URL, SECRET, QUIZFILE)
```

Cela fournit un tableau de résultats de la forme suivante
```
Res
```
:::{figure} doc_images/tableau_de_resultats.png
:name: resultats
:label:fig14
:alt: Tableau de résultats
:align: center
:width: 60%

Exemple de tableau de résultats. 
:::
que l'on pourra bien sûr exporter par exemple avec
```python
Res.to_csv("Résultats.csv")
```

### Correction pour un examen généré avec `quiz.exam_show()`

Pour un examen généré avec `quiz.exam_show()` qui est identifié par le titre donné à sa création, l'enseignant peut corriger spécifiquement les données recueillies sur le serveur, selon

(après avoir récupéré les données par `readData` et instancié un quiz -- étapes 1 et 3 précédentes),

```python
from labquiz.putils import getExamQuestions, getAllStudentsAnsvers, correctAll
exam_questions = getExamQuestions("Test pour voir", data)
students = exam_questions.keys()
students_answers = getAllStudentsAnsvers(students, data, maxtries=3)
correctAll(students_answers, quiz, data_filt, seuil=0, 
           exam_questions=exam_questions, weights=None, bareme=None, maxtries=3)
```

```python
from labquiz.putils import correctQuizzes

URL = "https://URL_UTILISÉE_POUR_RECUEILLIR_LES_RÉSULTATS"
SECRET = "MOT_DE_PASSE_SECRET_SPÉCIFIÉ_DANS_LE_SHEET"
QUIZFILE = "NOM_DU_FICHIER_DE_QUESTIONS.yml" #fichier de quiz CONTENANT les valeurs attendues

Res = correctQuizzes(URL, SECRET, QUIZFILE, title='Titre du test')
```
où l'on a précisé le titre du test à corriger par le paramètre `title`. 

:::{figure} doc_images/tableau_de_resultats_exam_show.png
:name: resultats
:label:fig15
:alt: Tableau de résultats
:align: center
:width: 80%

Exemple de tableau de résultats (`exam_show`). 
:::

### Options de correction

Quelques options supplémentaires peuvent être utilisées lors de la correction. 
```python
def correctQuizzes(URL, SECRET, QUIZFILE, title=None, seuil=0, weights=None, bareme=None, maxtries=1)
````
- title: si title n'est pas None, cela indique qu'il s'agit de la correction d'un test avec tirage au sort des questions de type `exam_show`, et dont le titre est title,
- seuil: seuil=0 seuille à zéros les notes de chaque question (sinon note négative possible) ; c'est la valeur par défaut, mais si on veut permetter des notes négatives par question, on peut la passer à -10 par exemple,
- weights: la matrice de poids (dictionnaire) dont on a déjà discuté dans [](calcul_du_score) 
- bareme:  poids des différentes questions dans le quiz. Si pas de barème, toutes les 
            questions sont au même poids pour le calcul de la note. Si poids d'une question 
            non spécifié, il est à 1 par défaut. Exemple: bareme = {'quiz3':4, 'quiz55':0} affecte un coefficient de 4 à la question quiz3 et neutralise la question quiz55 (toutes les autres questions auront un poids de 1),
- maxtries: Nombre d'essais permis. La correction s'effectue sur la dernière tentative inférieure ou égale à maxtries (et avant toute demande de correction, bouton "Corriger", si celui-ci est disponible).


### Dashboard

En cours de session, on peut donc récupérer le tableau des données, cf section précédente, et regarder les résultats. Les fonctions élémentaires suivantes permettent d'observer l'avancée globale 

```python
# Lecture des données
URL = "https://URL_UTILISÉE_POUR_RECUEILLIR_LES_RÉSULTATS"
SECRET = "MOT_DE_PASSE_SECRET_SPÉCIFIÉ_DANS_LE_SHEET"
data, data_filt = readData(URL, SECRET)
df_last = data_filt.drop_duplicates(
    subset=["student", "quiz_title"],
    keep="last")

#Le nombre de quizzes réalisés par étudiant
quiz_count_by_student = (
    df_last.groupby("student")["quiz_title"]
           .apply(len)
           .sort_index(ascending=False)
)
# Les scores par étudiant
score_by_student = (
    df_last.groupby("student")["score"]
           .apply(np.sum)
           .sort_index(ascending=False)
)
# Pour la classe complète
class_counts = (
    df_last["quiz_title"]
    .value_counts()
    .sort_index(
        key=lambda idx: idx.str.extract(r"(\d+)").astype(int)[0]
    )   # ordre alphabétique
)

#
print("Nombre de quizzes réalisés\n\n", quiz_count_by_student)
print("Scores obtenus\n\n", score_by_student)
print("Classe complète\n\n", class_counts)
```

On peut même mettre cela sous la forme d'un petit *dashboard* graphique que l'on rafraichira régulièrement. 

Le dashboard évoqué ci-dessus a finalement été mis en forme en une application dédiée, qui est décrite en [](quiz_dash). 

# Sécurité


## Principes 

Différentes mesures ont été prises afin d'assurer un niveau raisonnable d'identification des contributeurs, de respect des consignes et de limitation de la fraude. Parmi ces mesures, 

- Identification de la machine (via son système logiciel et matériel), ce qui fournit un identifiant utilisé dans toutes les transactions.
- Les paramètres et l'état du quiz sont conservés si on ré-instancie le quiz. Seul le redémarrage du noyau peut remettre les choses à zéro (et dans ce cas on perd aussi toutes ses données locales).
- Cryptage du fichier de questions : celui-ci est encrypté à l'aide d'une clé calculée à l'exécution et dépendant éventuellement d'une clé disponible sur un serveur distant, paramètre `mandatoryInternet=True`. 
- En mode connecté, qui est obligatoire si le paramètre `mandatoryInternet` a été positionné, *toutes* les validations, demandes de correction sont enregistrées et transmises, avec les réponses données au quiz courant, ce qui permet aussi la correction a posteriori
- Un hash des sources et paramètres est transmis, permettant de détecter des modifications des sources -- y compris monkey patching, ou des paramètres surveillés 
- Un *daemon* transmet également périodiquement l'état du système avec bien entendu l'identification de la machine. 

🧑‍🏫🏫 - ⚠️ - Ceci étant écrit, pour réaliser un *examen* dans de bonnes conditions, utilisez un fichier de questions SANS les réponses. Dès qu'il y a des réponses, même difficilement accessibles, un étudiant motivé arrivera toujours à les obtenir (et les partager). Si vous voulez évaluer lors d'un TP, soit ne diffuser que le fichier sans réponses, soit utiliser un tout petit coefficient. 


## 🧑‍🏫🏫 - Détecter des anomalies

Des fonctions ont été préparées pour ce faire. Elles permettent de tester l'intégrité au démarrage (`start`) ou au cours du temps, en isolant les modifications effectuées, ou en signalant que le hash n'est pas celui attendu ou a été modifié. Voici quelques exemples de mise en oeuvre 

### Intégrité
```python
from labquiz import QuizLab
from labquiz.putils import readData, getAllStudentsAnsvers, correctAll

## 1 - Lecture des données
URL = "https://URL_UTILISÉE_POUR_RECUEILLIR_LES_RÉSULTATS"
SECRET = "MOT_DE_PASSE_SECRET_SPÉCIFIÉ_DANS_LE_SHEET"
data, data_filt = readData(URL, SECRET)

## 2 - Students
students = sorted(list(data["student"].dropna().unique()))

## 3 - imports 
from labquiz.putils import  check_integrity_all_std, check_start_integrity_all_std, check_hash_integrity

## 4 - test

originalp = {'retries':2, 'exam_mode':True, 'test_mode':False}
starting_values = {'exam_mode':True, 'retries':2 }

print("------------------------")
print("==> test start_integrity")
print("------------------------")
check_start_integrity_all_std(starting_values, data)
print("--------------------------------")
print("==> test check_integrity_all_std")
print("--------------------------------")
check_integrity_all_std(originalp, students, data)
``` 
Les sorties sont les suivantes (tronquées avec l'indication [...])
```
------------------------
==> test check_integrity
------------------------
Dufour Léa: Modification machine pour le même nom d'étudiant
['21b1f154204eb9e7' '04b1f154204fa9e9']
Dufour Léa - enregistrement 13 :
      - Clé originale 'exam_mode' modifiée de True vers False
      - Clé originale 'test_mode' modifiée de False vers True
Dufour Léa - enregistrement 14 :
      - Clé originale 'exam_mode' modifiée de True vers False
      - Clé originale 'test_mode' modifiée de False vers True
Dufour Léa - enregistrement 15 :
      - Clé originale 'exam_mode' modifiée de True vers False
      - Clé originale 'test_mode' modifiée de False vers True
Dufour Léa - enregistrement 16 :
      - Clé originale 'exam_mode' modifiée de True vers False
      - Clé originale 'test_mode' modifiée de False vers True
[...]
------------------------
==> test start_integrity
------------------------
04b1f154204fa9e9 - enregistrement 42 : Clé originale 'exam_mode' modifiée de True vers False
04b1f154204fa9e9 - enregistrement 43 : Clé originale 'exam_mode' modifiée de True vers False
[...]
04b1f154204fa9e9 - enregistrement 53 : Clé originale 'exam_mode' modifiée de True vers False
21b1f154204eb9e7 - enregistrement 13 : Clé originale 'exam_mode' modifiée de True vers False
04b1f154204fa9e9 - enregistrement 27 : Clé originale 'exam_mode' modifiée de True vers False
--------------------------------
==> test check_integrity_all_std
--------------------------------
Dufour Léa: Modification machine pour le même nom d'étudiant
['21b1f154204eb9e7' '04b1f154204fa9e9']
Dufour Léa - enregistrement 13 :
      - Clé originale 'exam_mode' modifiée de True vers False
      - Clé originale 'test_mode' modifiée de False vers True
Dufour Léa - enregistrement 14 :
      - Clé originale 'exam_mode' modifiée de True vers False
      - Clé originale 'test_mode' modifiée de False vers True
Dufour Léa - enregistrement 15 :
      - Clé originale 'exam_mode' modifiée de True vers False
      - Clé originale 'test_mode' modifiée de False vers True
[...]
J Jean-Marc - enregistrement 54 :
      - Clé originale 'exam_mode' modifiée de True vers False
      - Clé originale 'test_mode' modifiée de False vers True
[...]
L Bobby - enregistrement 48 :
      - Clé originale 'exam_mode' modifiée de True vers False
      - Clé originale 'test_mode' modifiée de False vers True
L Bobby - enregistrement 49 :
      - Clé originale 'exam_mode' modifiée de True vers False
      - Clé originale 'test_mode' modifiée de False vers True
L Bobby - enregistrement 50 :
      - Clé originale 'exam_mode' modifiée de True vers False
      - Clé originale 'test_mode' modifiée de False vers True
L Bobby - enregistrement 51 :
      - Clé originale 'exam_mode' modifiée de True vers False
      - Clé originale 'test_mode' modifiée de False vers True
Lamarq Linda - enregistrement 30 :
      - Clé originale 'retries' modifiée de 2 vers 2000000
Lamarq Linda - enregistrement 31 :
      - Clé originale 'retries' modifiée de 2 vers 2000000
Lamarq Linda - enregistrement 32 :
      - Clé originale 'retries' modifiée de 2 vers 2000000
Lamarq Linda - enregistrement 33 :
      - Clé originale 'retries' modifiée de 2 vers 2000000
```

### Partage de machines
```python
"""
Démo : Détecte si une même machine a été utilisée pour plusieurs noms d'étudiants
"""
from labquiz.putils import check_machine
check_machine(data_filt) #Détecte si une même machine a été utilisée pour plusieurs noms d'étudiants
```
```
Même machine 04b1f154204fa9e9 utilisée par plusieurs étudiants
['Morane Bob' 'L Bobby' 'J Jean-Marc' 'Legrand John' 'Dufour Léa'
 'Makhoul Alain' 'Lamarq Linda']
````

### Contrôle du hash

Identification du hash attendu -- les paramètres `retries`, `exam_mode`, `test_mode` doivent être les mêmes que ceux imposés aux étudiants. 
```python

from labquiz.utils import get_full_object_hash

QUIZFILE = "NOM_DU_FICHIER_ORIGINAL" 
URL = ""
quiz = QuizLab(URL, QUIZFILE, needAuthentification=False, mandatoryInternet=False, retries=2, exam_mode=True)

wanted_hash = get_full_object_hash(quiz,  modules = ['main', 'utils'],
        WATCHLIST=['exam_mode', 'test_mode', 'retries'])

```
Et test pour tout le monde

```python
from labquiz.putils import check_hash_integrity
check_hash_integrity(data, 'full', wanted_hash=wanted_hash)
```
Ce qui fournit
```
⚠️ Dufour Léa, machine id 04b1f154204fa9e9 :
    👉🏼 Le source ou les paramètres ont été modifiés ou monkey patched
⚠️ index 17 hash: ddd8b0e5a3b35c41bb8db16cda874ff52af40a9ffdf56d7510abed65a9dec69f
⚠️ Dufour Léa, machine id 21b1f154204eb9e7 :
    👉🏼 Le source ou les paramètres ont été modifiés ou monkey patched
hash constatés :
fb85b7bceb28ecce4a047d1fb94428c60789505ebd89e22a5154628a125fd2e8 👍
ddd8b0e5a3b35c41bb8db16cda874ff52af40a9ffdf56d7510abed65a9dec69f ⚠️ index [13]

⚠️ J Jean-Marc, machine id 04b1f154204fa9e9 :
    👉🏼 Le source ou les paramètres ont été modifiés ou monkey patched
⚠️ index 54 hash: d5aa069407214f52b187ba479047d36fead12bc5b541ba3bceece6bc9f328490

⚠️ L Bobby, machine id 04b1f154204fa9e9 :
    👉🏼 Le source ou les paramètres ont été modifiés ou monkey patched
hash constatés :
880414219b2eadf47f234227c28927fc7d8a6f1a911bf40fa023f58d1a6cd83d ⚠️ index [43]
7de9df303f6cbbd183bcbc3745f4c2ed87d4e9a38afc79f8a47ea47981de43a5 ⚠️ index [45]
d5aa069407214f52b187ba479047d36fead12bc5b541ba3bceece6bc9f328490 ⚠️ index [50]

⚠️ Lamarq Linda, machine id 04b1f154204fa9e9 :
    👉🏼 Le source ou les paramètres ont été modifiés ou monkey patched
hash constatés :
fb85b7bceb28ecce4a047d1fb94428c60789505ebd89e22a5154628a125fd2e8 👍
4105fbe50a65562da899d9d9062a1853f8623c6b14b7f0a663cfc5244afd0b40 ⚠️ index [30]

⚠️ Morane Bob, machine id 04b1f154204fa9e9 :
    👉🏼 Le source ou les paramètres ont été modifiés ou monkey patched
⚠️ index 36 hash: b78c45342e260e62883a229b0b48364956e78b039ddbd0d64b6222d761ae8e4a
```

## Sécurité côté client

Comme écrit plus haut, si on veut noter les élèves ou réaliser un examen dans de bonnes conditions, la préconisation est d'utiliser un fichier de questions SANS les réponses. Comme le code source python est accessible, un étudiant ou un groupe d'étudiant pourra effectuer une étude, du reverse engineering et truquer les données envoyées au serveur. C'est raisonnablement complexe et difficile, mais cela ne peut être exclu. 

De ce fait, il faudrait ajouter une sécurité ``côté client'' avec une intervention non prédictible. Cela peut être le surveillant qui passe dans les rangs, ou l'exécution à la demande (par l'enseignant ou les élèves) d'un code spécifique masqué protégé par un mot de passe qui n'est révélé qu'à l'exécution. Ceci a été mis en place et devrait calmer pas mal les velléités de fraude. 

L'idée est donc de préparer un code, qui sera crypté à l'aide d'un mot de passe, exécuté à la demande, et qui effectuera un certain nombre de contrôles d'intégrité de la configuration de l'élève.

L'enseignant prépare son archive cryptée de la manière suivante :

```python
# example_create_secure_tar.py
# ⚠️ zutils.py doit être présent dans le répertoire local
from labquizdev.putils import create_secure_tar

# Install python_minifier si besoin
import importlib.util
import sys
import subprocess

module_name = "python_minifier"  
spec = importlib.util.find_spec(module_name)

if spec is None:
    print(f"{module_name} is not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", module_name])
import python_minifier

try:
    with open('zutils.py', 'r') as f:
            minified = python_minifier.minify(f.read(), remove_literal_statements=True, 
                            rename_globals=True, preserve_globals="do_the_check")
    with open('quiz_data/zutils.py', 'w') as f:
            f.write(minified)
except:
    pass
        
global_hash, src_hash = create_secure_tar(
    source_dir="quiz_data",
    output_file="quiz.tar.enc",
    password_open="MOT_DE_PASSE_POUR_ENCRYPTER__LARCHIVE",
    password_seal="SECOND_MOT_DE_PASSE_POUR_WATERMARQUER_LES_SOURCES"
)
```
:::{figure} doc_images/secure_tar.png
:name: secure_tar
:label:fig17
:alt: Create secure crypted tar
:align: center
:width: 60%

Résultat de l'exécution
:::

A l'intérieur de cette archive, le module zutils est complètement autonome, et contient une fonction `do_the_check`, personnalisable, qui est lancée automatiquement et effectue des contrôles, qui sont transmis au serveur. Dans la version fournie aux enseignants (mais personnalisable, répertoire `extras`), un *daemon* est lancé qui effectue le contrôle périodiquement. On y retrouvera le code d'intégrité des sources de zutils.py, watermarqué, sous la clé `session_hash`.

Côté élèves, il suffit d'appeler
``` 
quiz.check_quiz()
``` 
sur chaque machine, en utilisant le mote de passe `password_open`. 

:::{figure} doc_images/check_quiz.png
:name: check_quiz
:label:fig18
:alt: Check config integrity
:align: center
:width: 60%

Résultat de l'exécution
:::

(prepa_encode_file)= 
# 🧑‍🏫🏫 - Préparer et encoder le fichier de questions - `quiz_editor`

## Préparation manuelle et grands principes

Le fichier de questions est un fichier texte et on peut donc préparer et maintenir le fichier avec un simple éditeur de texte, en suivant la structure détaillée en [](structure_fichier_question). Une fois ce fichier préparé, on peut vouloir générer
- [enc] une version encodée base64 (pour éviter une consultation trop aisée)
- [crypt] une version cryptée avec une clé cachée
- [qo] une version ``questions only'' débarassée des tips et réponses

Notamment, 
- Pour un examen, vous distribuerez le fichier ``questions only'', encrypté ou pas (vous pouvez aussi parfaitement l'utiliser en dehors d'un examen, mais dans ce cas les élèves n'auront ni tips ni réponses, ni scores)
- Pour un mode test où vous prévoyez un *feedback* vers les élèves, utilisez la version avec réponses, en format encodé ou crypté, ajustez `retries` et imposez `test_mode=True` dans les paramètres passés à l'initialisation.

Une fonction permet d'effectuer ces différentes opérations :
```python
prepare_files(input_file, output_file, mode="crypt", pwd=""):    
    """
    Prepare YAML files for quizzes. 
    Outputs two files, with the basename given in `output_file`. 
    The second file is questions only and is the input stripped
    from responses ans tips. With the `mode="crypt"`, the input and stripped 
    versions are encrypted; with the `mode="enc"`, both files are base64 encoded; 
    finally, with `mode=yml`, files are not encoded nor encrypted.

    Parameters
    ----------
    input_file : str
        Path to the input YAML file.
    output_file : str
        Path to the output YAML file.
    mode : str, optional
        Mode to prepare the file. Choose from "crypt", "enc", or "yml".
    pwd : str, optional
        Password for file encryption.
```

Elle s'utilise par exemple de la manière suivante
```python
from labquiz.putils import prepare_files
prepare_files("quizzes_basic_filtering_test.yaml", "qbf.yml", mode="crypt", pwd=MOT_DE_PASSE)
```
ce qui produit
```
- Creating qbf_crypt.txt
- Creating qbf_qo_crypt.txt
⚠️ File crypted with pwd. Ensure to use the `mandatoryInternet=True` option in quiz init
```

➡️  Vous copierez le mot de passe éventuel dans la cellule A2 de la feuille Config du google-sheet, cf [](google-sheet), si vous utilisez un gs pour recueillir les résultats. 


- Pour un examen, vous distribuerez le fichier ``questions only'', encrypté ou pas (vous pouvez aussi parfaitement l'utiliser en dehors d'un examen, mais dans ce cas les élèves n'auront ni tips ni réponses, ni scores)
- Pour un mode test ou vous prévoyez un *feedback* vers les élèves, utilisez la version avec réponses, en format encodé ou crypté, ajustez `retries` et imposez `test_mode=True` dans les paramètres passés à l'initialisation.


## Préparation à l'aide de `quiz_editor`

La préparation manuelle est possible pour de petites bases, mais un éditeur spécifique a également été créé pour permettre l'édition et différentes opérations (extraction, encodage, cryptage, conversion vers d'autres formats. 

Cet éditeur est disponible ici :

https://jfb-quizeditor.streamlit.app/

Il permet d'éditer les fichiers dont on a décrit la structure et possède quelques fonctionnalités supplémentaires :

- possibilité de définir des {underline}`catégories` et des {underline}`tags`, et de sélectionner les questions en filtrant sur catégories et tags,
- extraction et sauvegarde d'une partie des questions,
- préparation d'une version encodée ou cryptée (comme décrit ci-avant),
- <mark>conversion au format AMC-$\LaTeX$ </mark>, de sorte à pouvoir réutiliser les questions dans un QCM papier AMC, (Les catégories sont utilisées pour définir le type `\element`), un exemple de sortie [ici](https://www.esiee.fr/~bercherj/labquizDemo/demo_export/demo_export.tex)
- <mark>conversion au format HTML</mark> 
 avec réponses intégrées, de sorte à pouvoir créer une page web «d'auto-évaluation» -- un exemple de résultat d'export ici [https://www.esiee.fr/~bercherj/labquizDemo/files/demo_export/demo_export_train.html](https://www.esiee.fr/~bercherj/labquizDemo/files/demo_export/demo_export_train.html)
- conversion au format HTML **examen**, sans les réponses, et avec soumission en temps réel des résultats à un google-sheet, avec correction a posteriori, comme décrit en [](calcul_du_score_correction) ou à l'aide du dashboard comme décrit plus loin [](quiz_dash)  -- exemple ici [https://www.esiee.fr/~bercherj/labquizDemo/files/demo_export/demo_export_exam.html](https://www.esiee.fr/~bercherj/labquizDemo/files/demo_export/demo_export_exam.html)

:::{figure} doc_images/quiz_editor_2.png
:name: quiz_editor_1
:label:fig24
:alt: quiz_editor
:align: center
:width: 90%

`quiz_editor` -- édition d'une question, avec catégorie, tags, choix du type de question (qcm, numeric, etc)
:::

:::{figure} doc_images/quiz_editor_1.png
:name: quiz_editor_2
:label:fig25
:alt: quiz_editor
:align: center
:width: 90%

`quiz_editor` -- édition d'une proposition -- correcte ou non, indice (tip), réponse (affichée lors de la correction), bonus, malus...
:::

:::{figure} doc_images/exports.png
:name: quiz_editor_exports
:label:fig26
:alt: quiz_editor_exports
:align: center


`quiz_editor` -- Exemples d'exports (YAML, AMC-$\LaTeX$, HTML-interactif, HTML-exam)
:::

(quiz_dash)=
# 🧑‍🏫🏫  - Suivre en temps réel et corriger avec `quiz_dash`

Comme décrit précédemment, on peut charger le tableau de résultats au niveau du terminal et effectuer tous les tests et statistqies que l'on veut. Cependant, il est plus aisé d'utiliser un petit utilitaire graphique pour ce faire. Ce dashboard de suivi et exploitation est disponible ici :

https://jfb-quizdash.streamlit.app/

À partir de la spécification de l'URL du google-sheet, du mot de passe de lecture associé et du fichier YAML contenant les réponses, ce dashboard permet de : 

- <mark>suivre au cours du temps</mark>, avec un taux de rafraichissement réglable, les soumissions effectuées par chaque participant, avec les labels des quizzes concernés,
- <mark>contrôler l'intégrité</mark>, c'est à dire vérifier que les paramètres (nombre d'essais autorisés, mode, etc) n'ont pas été modifiés, vérifier le hash des sources, de l'objet en mémoire et de ses dépendances, 
- <mark>visualiser</mark>, au cours du temps, la progression de chaque participant (filtrable) et du groupe complet,
- <mark>corriger</mark> et récupérer le tableau des résultats, 
- avec la possibilité d'ajuster la matrice de poids (pour les qcm) et le barème par question...

Quelques copies d'écran d'un suivi réel :

:::{figure} doc_images/dash_parameters.png
:name: dash_parameters
:label:fig26
:alt: quiz_dash
:align: center
:width: 90%

`quiz_dash` -- Entrée des paramètres de configuration monitoring/correction
:::

:::{figure} doc_images/Monitoring_integrity.png
:name: Monitoring_integrity
:label:fig27
:alt: quiz_dash
:align: center
:width: 90%

`quiz_dash` -- Monitoring de l'intégrité
:::

:::{figure} doc_images/Monitoring_quizzes.png
:name: Monitoring_activity
:label:fig28
:alt: quiz_dash
:align: center
:width: 90%

`quiz_dash` -- Monitoring des quizzes réalisés par les élèves, le groupe. Rafraichissement automatique possible et ajustable (les noms des élèves ont été masqués) 
:::

:::{figure} doc_images/Monitoring_marks.png
:name: Monitoring_marks
:label:fig29
:alt: quiz_dash
:align: center
:width: 90%

`quiz_dash` -- Correction automatisée, avec possibilité d'ajuster le berème (recalcul automatique) ; (les noms des élèves ont été masqués). Bien sûr, le tableau de résultats est téléchargeable.  
:::

(google-sheet)= 
# 🧑‍🏫🏫 - Créer son google sheet pour recueillir les résultats

Pour recueillir les données, on peut simplement utiliser un google sheet. Dans une version ultérieure, un petit serveur Flask pourrait être utilisé, mais il faut le déployer quelque part. Pour le moment une version google sheet est fonctionnelle et aisée à mettre en oeuvre. 

## Le plus simple du plus simple...

```{hint} Hint:  Le plus simple du plus simple
**Le plus simple du plus simple** est d'utiliser une copie du modèle qui a été préparé à cet effet :
pour ce faire, cliquer sur le lien suivant, 


https://docs.google.com/spreadsheets/d/1-hDtosDAA3ehy4iqFGU5D5NQirGY3c-HxLDmnsoG6AU/edit?usp=sharing

- puis menu Fichier/Créer une copie (renommez le fichier),
- et déployer par : menu Extensions/Apps Script et cliquer déployer en haut à droite ; choisir `Application Web`, partager avec tout le monde.

Copier et conserver le lien `https://script.google.com/macros/.../exec`, c'est l'URL que vous utiliserez par la suite (à diffuser aux élèves et à utiliser pour lire des données recueillies). Vous pourrez retrouver cette dresse en consultant "Gérer vos déploiement". 

🎉 Enregistrer et c'est bon !

Vous pouvez parcourir ce qui suit si vous voulez créer le google sheet par vous même ou comprendre la signification des paramètres de la feuille Config; Il vous faut ajuster les valeurs de Pwd sur la feuille Config, la valeur de SECRET dans le code AppScript (Extensions/Déployer, 1ere ligne). Voir les points 4, 5 et 6 ci-dessous. 
```

- - -

## Préparation manuelle

- 1 - Créer un google sheet à l'aide de votre compte
- 2 - Sur la première ligne de la première feuille, insérer les données suivantes, qui serviront de header à votre tableau :
```
timestamp	send_timestamp	notebook_id	student	quiz_title	event_type	parameters	answers	score
```

:::{figure} doc_images/1ereLigneSheet.png
:name: dashboard
:label:fig16
:alt: 1ereLigneSheet.png
:align: center
:width: 90%

Première ligne du sheet.
:::
- 3 - Créer une nouvelle feuille en appuyant sur le `+` puis la renommer en Config (par CTRL + clic sur l'onglet correspondant).

:::{figure} doc_images/creationfeuilleConfig.png
:name: dashboard
:label:fig17
:alt: creationfeuilleConfig
:align: center
:width: 30%

Création de la feuille `Config`.
:::

- 4 - Dans la feuille `Config`, créer les données suivantes

```
Pwd	Réception des données
Wrktz	TRUE
	
NMAX	NKEEP
2000	2500
````

:::{figure} doc_images/contenuFeuilleConfig.png
:name: dashboard
:label:fig18
:alt: contenuFeuilleConfig
:align: center
:width: 30%

Données de la feuille `Config`.
:::

👉🏼 Pwd est la clé qui sera utilisée pour contrôler la bonne connexion avec la feuille et participer à l'encryptage du fichier de questions. Modifier la valeur de la clé en A2 et gardez là en mémoire. NKEEP est le nombre de lignes maximum conservées dans la feuille. Le seuillage s'effectue dès que le nombre de lignes dépasse NMAX. 

👉🏼  Dans la feuille Config, sélectionner la cellule `B2`. Aller dans le menu `Données`, choisir `Validation des données`. Puis dans `Règle de validation des données`, choisir `case à cocher`. Assurez vous que la case soit cochée, sinon vous ne recevrez rien ! 

⚠️ *Et donc pour interrompre la réception de données, décocher* ! --> Cela peut être utile car certains laissent tourner les choses sur leur ordinateurs, et comme il y a un `check_alive` de vérification d'intégrité envoyé périodiquement, cela peut remplir le google-sheet (bien qu'on ait installé une limite max). 

- 5 - Dans le menu `Extensions`, cliquer sur `Apps Script`. Dans l'onglet qui s'ouvre, nommer votre projet, puis dans la page de code, après avoir effacé ce qui s'y trouve, coller le code joint dans le fichier code_gs.txt joint (dossier `extras`) et remplacer la valeur de `SECRET` sur la première ligne. Si jamais votre première feuille ne s'appelle pas `Feuille 1`, renommez la ou modifiez la constante sur la 3e ligne du code. 

:::{figure} doc_images/ExtensionsAppsScript.png
:name: dashboard
:label:fig19
:alt: ExtensionsAppsScript
:align: center
:width: 60%

Créer l'extension pour entre le code google script.
:::
- 6 - En haut à droite, cliquer sur `Déployer`puis `Nouveau déploiement`, choisir le type `Application Web`, partager avec tout le monde, copier le lien `https://script.google.com/...`, c'est l'URL que vous utiliserez par la suite (à diffuser aux élèves et à utiliser pour lire des données recueillies). Vous pourrez retrouver cette dresse en consultant "Gérer vos déploiement". 

🎉 Enregistrer et c'est bon !

:::{figure} doc_images/deployments.png
:name: dashboard
:label:fig20
:alt: gérerLesDéploiements
:align: center
:width: 60%

Déployer !
:::


(jupyterlite)= 
# 🧑‍🏫🏫 - Déployer sur Jupyterlite

[Jupyterlite](https://jupyterlite.readthedocs.io/en/stable/index.html) est une distribution de Python/Jupyterlab qui tourne entièrement dans le navigateur. Les différents packages python ont été portés en WebAssembly (WASM), un format binaire qui est exécutables dans les navigateurs. Pour des bibliothèques optimisées (numpy, pandas, ...) les performances sont équivalentes aux performances natives. Pour du pur python (boucles, etc), l'exécution est 3 à 10 fois plus lente. Ce que cela signifie néanmoins, c'est qu'on peut préparer des programmes python et des notebooks Jupyter qui sont entièrement exécutables dans un navigateur, avec *zéro installation* !  Et cela fonctionne indépendamment du système, et même sur tablette, téléphone...

Vous trouverez par exemple ici : [https://perso.esiee.fr/~bercherj/JliteNotes/lab/](https://perso.esiee.fr/~bercherj/JliteNotes/lab/) un exemple de TP distribué de cette manière, et intégrant labquiz. 

La préparation et le déploiement sous Jupyterlite ne sont pas exactement immédiats (il y a des docs à suivre et des adaptations à faire), aussi je vous propose un raccourci qui est celui-ci :
1. Télécharger l'archive d'une version déployée, disponible ici <br>
https://drive.google.com/file/d/14BnzVmPO6I8uOMEmNC6BTMsjXL3RLrQL/view?usp=sharing 
<br> et la décompresser sur votre disque, 

2. Placer les fichiers que vous souhaitez mettre à disposition dans `_output/files`
3. Exécuter le programme `update_jupyterlite_contents.py` par `python update_jupyterlite_contents.py`. Installer les dépendances si nécessaire. 
> 👉🏼 Celui-ci va mettre à jour les listes et caractéristiques des fichiers dans la distribution
4. **init**) Téléverser sur votre compte web, typiquement sur `login/public_html`, *la totalité* du contenu de `_output`, dans un répertoire tel que `MonJliteAMoi`, et vous obtiendrez une distribution accessible et exécutable sous `https://www.esiee.fr/~login/MonJliteAMoi`. Si vous ne savez pas faire, adressez vous à votre gourou informatique local le plus proche. 
5. **maj**) Si vous ne devez qu'effectuer une mise à jour, il vous suffira de copier le contenu de `_output/files` **et** de `_output/api` vers `login/public_html/MonJliteAMoi/files` et `login/public_html/MonJliteAMoi/api`. 

Pour les mises à jour suivantes, ajouter ou modifier des fichiers, vous n'aurez qu'à exécuter les étapes 2, 3, et 5.maj) ; ce qui vous économisera le téléversement des 70 Mo de l'étape 4.init). 

# Démonstration et exercices

Explorez le notebook de démonstration `labQuizDemo.ipynb` et expérimentez. 
Si vous voulez aller plus loin, et si vous êtes arrivé jusque là c'est probablement que vous êtes motivé, ci-dessous quelques pistes.  


- **Exercice** : créer un google sheet de nom `MonPremierGSQuiz` en suivant les instructions de la [](google-sheet).  Changer le mot de passe de lecture en 1er ligne du script accessible par Extensions/Apps script, puis déployer. Noter l'adresse du déploiement et le mot de passe. 

- **Exercice** : Chargez, ou créez une base de question dans `quiz_editor`. Ajouter quelques questions. Sauvegarder le nouveau fichier. Exporter une version HTML, une version AMC. Mettre en ligne le HTML sur votre compte et tester (ou tester en local).

- **Exercice** : Exporter une version encryptée de vos questions (notez le mot de passe et mettez le à jour dans le google-sheet, page Config).

- **Exercice** : Créer un notebook Jupyter, importer labquiz (comme dans l'entête de ce fichier ou dans le notebook de démo -- adaptez éventuellement le numéro de version en consultant https://github.com/jfbercher/labquiz), instancier un quiz, -- voir [](utilisation), intégrer quelques questions et tester.

- **Exercice** : Instancier un quiz lié à l'URL de votre google sheet, intégrez quelques questions dans votre notebook et vérifiez que les réponses apparaissent dans le google sheet. 

- **Exercice** : Utilisez `quiz_dash`pour lire les résultats. Changez les paramètres dans votre notebook (passage de `mode_examen` de True à False, modification de `retries` et voyez si ceci lève une alerte. Lancez la correction sur vos quelques essais. 

- **Exercice** : Créer une version jupyterlite comme expliqué en [](jupyterlite), ajoutez vos fichiers, téléversez cela sur votre compte info dans `public_html` et vérifiez en vous connectant à l'adresse web correspondante.  




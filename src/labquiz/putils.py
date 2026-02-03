import yaml
import random
import sys, io
from io import StringIO
from pathlib import Path
import copy
import re
import numpy as np
import pandas as pd
import requests
import ipywidgets as widgets
from IPython.display import display, HTML
from datetime import datetime
import hashlib, base64, json
import tarfile
from cryptography.fernet import Fernet, InvalidToken

qz = ('prep '*3).strip().split(' ')
qz.extend(['_get_protected_data', 'show', 'QuizLab'])

def shuffle_quiz_propositions(input_file, output_file):
    # Lecture du fichier YAML
    with open(input_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Mélange des propositions pour chaque quiz
    for quiz_name, quiz_content in data.items():
        if "propositions" in quiz_content:
            random.shuffle(quiz_content["propositions"])

    # Écriture dans un nouveau fichier
    with open(output_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False
        )


def encode_dict_base64(obj: dict) -> str:
    json_str = json.dumps(obj, ensure_ascii=False)
    return base64.b64encode(json_str.encode("utf-8")).decode("ascii")
    

def encode_file(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    for quiz in data.values():
        if "propositions" in quiz:
            quiz["propositions"] = [
                encode_dict_base64(p) for p in quiz["propositions"]
            ]
        if "constraints" in quiz:
            quiz["constraints"] = [
                encode_dict_base64(p) for p in quiz["constraints"]
            ]

    with open(output_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        
def crypt_file(input_file, output_file, pwd='', verbose=False):
    
    ctx = qz + [output_file, pwd]
     
    with open(input_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    if "title" in data.keys(): 
        key = b'zzFJ0WBRXuFC1qJOFGxeqt3NUNcFR9vdZqFkCry-DYw='
        frt = Fernet(key)
        data['title'] = frt.encrypt(data['title'].encode("utf-8") )
    
    for quiz_id, quiz in data.items():
        if quiz_id == "title": continue
        nctx = ctx + [quiz_id]
        fingerprint = "-".join(nctx).encode()
        #print("fingerprint", fingerprint)
        hash_digest = hashlib.sha256(fingerprint).digest()
        key = base64.urlsafe_b64encode(hash_digest)
        frt = Fernet(key)
        data[quiz_id] = frt.encrypt(json.dumps(quiz).encode("utf-8") ) 
        if verbose: print(quiz)
               

    with open(output_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def decrypt_file(input_file, output_file=None, pwd='', verbose=False):
    
    ctx = qz + [output_file, pwd]
    #ctx = ['prep', 'prep', 'prep', '_get_protected_data', 'show', 'QuizLab']
    fingerprint = "-".join(ctx).encode()
    hash_digest = hashlib.sha256(fingerprint).digest()
    key = base64.urlsafe_b64encode(hash_digest)
    frt = Fernet(key)
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "title" in data.keys(): 
        key = b'zzFJ0WBRXuFC1qJOFGxeqt3NUNcFR9vdZqFkCry-DYw='
        frt = Fernet(key)
        data['title'] = frt.decrypt(data['title'].decode("utf-8") )
         
    for quiz_id, quiz in data.items():
        if quiz_id == "title": continue
        nctx = ctx + [quiz_id]
        fingerprint = "-".join(nctx).encode()
        #print("fingerprint", fingerprint)
        hash_digest = hashlib.sha256(fingerprint).digest()
        key = base64.urlsafe_b64encode(hash_digest)
        #print("key", key)
        frt = Fernet(key)
        
        quiz = json.loads(frt.decrypt(quiz).decode('utf-8')) 
        data[quiz_id] = quiz

        if verbose: 
            print(quiz)
            
    if output_file is not None:
        with open(output_file, "w", encoding="utf-8") as f:
             yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
#decrypt_file("quizzes_basic_filtering_test.enc", verbose=True)
        
        
def quiz_propositions_for_exam(input_file, output_file):
    # Entrée: fichier de questions initial ou mélangé
    # Sortie : Fichier sans les solutions, les réponses ni les tips
    # Lecture du fichier YAML
    with open(input_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # suppression indices pour chaque quiz
    for quiz_name, quiz_content in data.items():
        if quiz_name == "title": continue 
        quiz_content.pop("constraints", None)
        for prop in quiz_content["propositions"]:
            '''
            prop["expected"] = ""
            prop["reponse"] = ""
            prop["tip"] = ""
            '''  
            keys_to_remove = {"expected", "reponse", "tip"}
            for k in keys_to_remove:
                prop.pop(k, None)
            
    # Écriture dans un nouveau fichier
    with open(output_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False
        )


def quiz_to_dict_of_solutions(input_file):
    """extrait les solutions du fichier initial
    
    Parameters
    ----------
        input_file:   str
            fichier de questions initial ou mélangé    
    Returns
    -------
        out :         dict
            dictionnaire {quiz1: {label1:expected, label2:expected, ...}, 
            quiz2:{label1:expected, label2:expected, ...} }
    """
    # Lecture du fichier YAML
    with open(input_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 
    dico = {quiz_name:  {prop["label"]:prop["expected"] for prop in quiz_content["propositions"]} 
            for quiz_name, quiz_content in data.items()}
    out = {l: dict(sorted(d.items(), key=lambda item: item[0]))
                       for l,d in zip(dico.keys(), dico.values())}
    # On trie sur les clés, on ne sait jamais !
    
    return out
        

def crypt_data(data, output_file, pwd=''):
    
    data = copy.deepcopy(data)
    ctx = qz + [output_file, pwd]
    if "title" in data.keys(): 
        key = b'zzFJ0WBRXuFC1qJOFGxeqt3NUNcFR9vdZqFkCry-DYw='
        frt = Fernet(key)
        data['title'] = frt.encrypt(data['title'].encode("utf-8") )
    
    for quiz_id, quiz in data.items():
        if quiz_id == "title": continue
        nctx = ctx + [quiz_id]
        fingerprint = "-".join(nctx).encode()
        hash_digest = hashlib.sha256(fingerprint).digest()
        key = base64.urlsafe_b64encode(hash_digest)
        frt = Fernet(key)
        data[quiz_id] = frt.encrypt(json.dumps(quiz).encode("utf-8") ) 
    return data


def encode_data(data):
    data = copy.deepcopy(data)
    for quiz in data.values():
        if "propositions" in quiz:
            quiz["propositions"] = [
                encode_dict_base64(p) for p in quiz["propositions"]
            ]
        if "constraints" in quiz:
            quiz["constraints"] = [
                encode_dict_base64(p) for p in quiz["constraints"]
            ]
    return data


def prepare_files(input_file, output_file, mode="crypt", pwd=""):    
    """
    Prepare YAML files for quizzes. 
    Outputs two files, with the basename given in `output_file`. 
    The second file is questions only and is the input stripped
    from responses ans tips. With the `mode="crypt"`, the input and stripped 
    versions are encrypted; with the `mode="enc"`, both files are binhex encoded; 
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

    Returns
    -------
    None
    """    
    import copy
    # Lecture du fichier YAML
    with open(input_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Mélange des propositions pour chaque quiz
    for quiz_name, quiz_content in data.items():
        if "propositions" in quiz_content:
            random.shuffle(quiz_content["propositions"])
            
    # Questions only
    data_only = copy.deepcopy(data)
    # suppression indices pour chaque quiz
    for quiz_name, quiz_content in data_only.items():
        if quiz_name == "title": continue 
        quiz_content.pop("constraints", None)
        for prop in quiz_content["propositions"]:
            keys_to_remove = {"expected", "reponse", "tip"}
            for k in keys_to_remove:
                prop.pop(k, None)
  
    path = Path(output_file)
    if mode == "crypt":
        data_out = crypt_data(data, f"{path.stem}_crypt.txt", pwd=pwd)
        data_only_out = crypt_data(data_only, f"{path.stem}_qo_crypt.txt",  pwd=pwd)
    elif mode == "enc":
        data_out = encode_data(data)
        data_only_out = encode_data(data_only) 
    else:
        mode = "yml"
        data_out = data
        data_only_out = data_only
        
    for suffix, data in zip([mode, f"qo_{mode}"], [data_out, data_only_out]):
        out = path.with_name(f"{path.stem}_{suffix}.txt")
        
        with out.open("w", encoding="utf-8") as f:
            print(f"- Creating {out}")
            yaml.safe_dump(
                data,
                f,
                allow_unicode=True,
                sort_keys=False
            )
    if mode == "crypt" and pwd != '':
        print("⚠️ File crypted with pwd. Ensure to use the `madatoryInternet=True` option in quiz init")    
    
    
#=====================================================
# Lecture des données et correction
#=====================================================

def readData(URL, SECRET):
    """Lecture des données
    
    readData(URL, SECRET)
    return df, df_last
    Se connecte pour récupérer le tableau de données désigné par l'adresse URL et protégé par SECRET
    
    Parameters
    ----------
        URL:    str
                Chemin d'accès
        SECRET: str
                Chaîne utilisée pour accéder aux données
    Returns
    -------
        df:     pandas dataframe
                tableau complet
        df_filt:pandas dataframe
                tableau filtré juste sur les validations et corrections
    """
    
    #global df, df_last

    try:
        r = requests.get(URL, params={"secret": SECRET})
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text)) 
        df.student = df.student.apply(lambda s: s.strip().title() if isinstance(s, str) else s)  #Normalisation colonne student
        df["answers"] = df.answers.apply(lambda x: parse_custom_dict(x)) #Décodage de la colonne answers
        df["parameters"] = df.parameters.apply(lambda x: parse_custom_dict(x)) #Décodage de la colonne answers
        now = datetime.now().strftime("%H:%M:%S")

        footer_msg = f"<span style='color:green; font-weight:bold'>✔ Sheet chargé ... ({len(df)} lignes) - {now} </span>"
        display(HTML(footer_msg))

    except Exception as e:
        footer_msg =  f"<span style='color:red; font-weight:bold'>✘ Erreur de chargement {e}</span>"
        display(HTML(footer_msg))
        
    df = df.sort_values("timestamp")
    #df_last = df.drop_duplicates(
    #    subset=["student", "quiz_title", "event_type"],
    #    keep="last"
    #)
    #df_last_validate or correction 
    #df_last = df_last.query("event_type == 'validate' ")
    df_last = df.query("event_type in ['validate', 'validate_exam', 'correction'] ")
    return df, df_last

def convert_ans(answers):
    """Conversion des réponses (str) en dictionnaire
    
    convert_ans(answers)
    return dico
    Convertit les réponses (string) en dictionnaire des réponses
    
    Parameters
    ----------
        answers:    str
    Returns
    -------
        dico:       dict
                    dictionnaire correspondant
    """
    try:
        dico = {k: v.lower() == "true" for k, v in (item.rsplit("=",1) for item in answers.strip("{}").split(", "))}
        dico = dict(sorted(dico.items(), key=lambda item: item[0])) # On trie les clés, on ne sait jamais
    except:
        print(answers)
    return dico


def correct_ans(quiz, quiz_id, given, weights=None, constraints=None):
    """Correction d'une réponse
    
    # correct_ans(quiz, quiz_id, given, weights=None)
    # return score, score_max
    Corrige les réponses données dans `given` en comparant avec celles contenues
    dans le sujet (avec corrigé) pour le label `quiz_id`. 
    
    Parameters
    ----------
        quiz:      objet 
            Objet QuizLab dans lequel a été chargé le sujet
            >> quiz = QuizLab(URL, QUIZFILE, needAuthentification=False, retries=100)
        quiz_id:   str
            Clé du quiz voulu
        given:     dict 
            Réponses données
        weights:   pandas dataframe
            Matrice de poids 
            ex:         default_weights = {
                (True, True):   1,  # Vrai Positif
                (True, False): -1,  # Faux Positif 
                (False, True):  0,  # Faux Négatif (oubli)
                (False, False): 0   # Vrai Négatif
                }
    Returns
    -------
        score:     float
                   score obtenu
        score_max: float
                   score max pouvant être atteint
    """
    from .utils import calculate_quiz_score
    question, quiz_type, propositions, constraints = quiz._QuizLab__load_quiz(quiz_id)
    #NB: Le nom `_QuizLab__load_quiz` correspond à la méthode dunder __loadquiz dans main
    propositions.sort(key=lambda d: d["label"]) # tri sur les clés
    given = {k:given[k] for k in sorted(given)}
    #print(given, propositions)
    return calculate_quiz_score(quiz_type, given, propositions, weights=weights, constraints=constraints)

    


def check_ans(student, quiz_id, df, maxtries=3):
    """Sélectionne la réponse à utiliser
    
    check_ans(student, quiz_id, df, maxtries=maxtries)
    return given
    Sélectionne la "bonne" réponse : 
       - celle de rang <= maxtries 
       - si correction demandée : la dernière avant correction si <= maxtries sinon maxtries 
    
    Arguments
    ---------
        student:   str
            nom de l'étudiant à rechercher dans le tableau
        quiz_id:   str
            id du string à utiliser (à rechercher dans le tableau)
        df:        pandas dataframe
            tableau de données
        maxtries:  int
            Nombre d'essais max
    Returns
    -------
        given:     dict
            dictionnaire de la réponse
            
    
    """
    extract = df.query(f"student == '{student}' & quiz_title=='{quiz_id}'").reset_index()
    if len(extract):
        correction = extract[extract['event_type'] == 'correction'].index
        if len(correction):
            idx = (correction[0]-1) if (correction[0]-1) <= maxtries else maxtries
        else: 
            idx = len(extract)-1 if (len(extract)<=maxtries) else maxtries
        
        if (idx==-1): 
            #print(quiz_id, "A demandé la correction avant valid")
            return {}
        #quiz_test = extract.loc[idx]["answers"]
        #given = convert_ans(quiz_test)
        given = extract.loc[idx]["answers"]
        #print(extract.loc[idx]["answers"])
    else:
        given = {}
    return given   



def new_parse_custom_dict(text):  # Changement de format !! Bien plus simple
    """Convertit la chaîne de caractères `answers`en dictionnaire
    
    parse_custom_dict(text)
    return result
    Avec le changement de format, juste un décodage json
    
    Parameters
    ----------
        text:    str
            chaîne de caractère au format '{"val1":true, "val2":false, "val3":float, "val4":"{...}"}'
    Returns
    -------
        result:  dict
            dictionnaire trié des clés:valeurs identifiées       
    """
    import json
    d = json.loads(text)
    if isinstance(d, dict):
        d_trie = {k: v for k, v in sorted(d.items())}
        return d_trie
    return d

def old_parse_custom_dict(text):
    """Convertit une chaîne de caractères en dictionnaire
    
    parse_custom_dict(text)
    return result
    Convertit une chaîne de caractère au format "{val1=true, val2=false, val3=float, val4={...}}"
    en un dictionnaire contenant des valeurs booléennes, des nombres ou d'autres dictionnaires
    
    Parameters
    ----------
        text:    str
            chaîne de caractère au format "{val1:true, val2=false, val3=float, val4={...}}"
    Returns
    -------
        result:  dict
            dictionnaire des clés:valeurs identifiées
    
    # Merci Gemini !!         
    """
    # Nettoyage des accolades extérieures
    text = text.strip().strip("{}")
    result = {}
    
    # 1. Découpage intelligent par virgule (ignore les virgules dans les { })
    parts = []
    bracket_level = 0
    current = []
    for char in text:
        if char == '{': bracket_level += 1
        elif char == '}': bracket_level -= 1
        
        if char == ',' and bracket_level == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current).strip())

    # 2. Analyse de chaque partie
    for part in parts:
        if not part: continue
        
        # On cherche le DERNIER signe '=' qui sépare la clé de la valeur.
        # La valeur peut être :
        # - Un bloc entre accolades { ... }
        # - Une chaîne (hash, texte, booléen, nombre)
        if '=' in part:
            # On découpe par la droite pour gérer "Sinc x=0 Handling=false"
            # On cherche le '=' qui précède une valeur type (bool, num, hash, ou {)
            match = re.search(r'^(.*)=(.*)$', part)
            if match:
                key = match.group(1).strip()
                val_raw = match.group(2).strip()
                
                # Conversion des types
                if val_raw.startswith('{'):
                    result[key] = parse_custom_dict(val_raw)
                elif val_raw.lower() == 'true':
                    result[key] = True
                elif val_raw.lower() == 'false':
                    result[key] = False
                else:
                    try:
                        # On tente le nombre, sinon on garde la chaîne (hash, etc.)
                        if '.' in val_raw:
                            result[key] = float(val_raw)
                        else:
                            result[key] = int(val_raw)
                    except ValueError:
                        result[key] = val_raw # C'est ici que les hash sont sauvés
                        
        result = dict(sorted(result.items(), key=lambda item: item[0])) # On trie les clés, on ne sait jamais
                         
    return result

def parse_custom_dict(text):
    """Convertit la chaîne de réponses dans df['answers'] en dictionnaire, clés triées
    """
    if pd.isnull(text): return text
    try: 
        result = new_parse_custom_dict(text)
    except Exception as e:
        #print(f"Erreur level 1 décodage parse_custom_dict() pour {text}")
        #print("Erreur level 1", e)
        try:
            result = old_parse_custom_dict(text)
        except Exception as e:
            print(f"Erreur level 2 décodage parse_custom_dict() pour {text}")
            print("Erreur level 2", e)
            return text
    return result



def correctAll(students_answers, quiz, df, seuil=0, exam_questions=None, weights=None, bareme=None, maxtries=3):
    """Corrige toutes les réponses
    
    correctAll(students, sols, df, seuil=0, Poids=None, bareme=None)
    return Res 
    Corrige l'ensemble des résultats des étudiants de `students` à partir des solutions `sols`
    
    Arguments
    ---------
        students:   dict
            Réponses données
        quiz:      objet 
            Objet QuizLab dans lequel a été chargé le sujet
            >> quiz = QuizLab(URL, QUIZFILE, needAuthentification=False, retries=100)
        df:         pandas dataframe
            Tableau d'entrée
        seuil:      float
            seuil=0 seuille à zéros les notes de chaque question (sinon note négative possible)
        exam_questions: list (default None)
            liste des questions à corriger (par défaut toutes)
        weights:    dict
            Matrice de poids 
            ex: default_weights = {
                (True, True):   1,  # Vrai Positif
                (True, False): -1,  # Faux Positif 
                (False, True):  0,  # Faux Négatif (oubli)
                (False, False): 0   # Vrai Négatif
            }
        bareme:     dict
            Barême = poids des différentes questions dans le quiz. Si pas de barême, toutes les 
            questions sont au même poids pour le calcul de la note. Si poids d'une question 
            non spécifié, il est à 1 par défaut. ex: bareme = {'quiz3':4, 'quiz55':0} (tous les autres 
            avec un poids de 1).
        maxtries:   int
            Nombre d'essais permis
        
    Returns
    -------
        Res:       pandas dataframe
            lignes : étudiants (`students`), colonnes : chacun des quizzes + une colonne Note 

    """
    import pandas as pd
    
    allquizzes = [q for q in quiz.quiz_bank.keys() if q != "title"]
    students = students_answers.keys()
    if exam_questions is None: exam_questions = {s:allquizzes for s in students}
    Res = pd.DataFrame(index=students, columns=allquizzes)
    if bareme is not None: 
        bareme = {q:1 if q not in bareme.keys() else bareme[q] for q in allquizzes }
    else:
        bareme = {q:1 for q in allquizzes}
    for s in students:
        for quiz_id in exam_questions[s]:
            given = students_answers[s].get(quiz_id, {})
            if given == {}: continue
            score, score_max = 0,1   # précaution            
            try:
                score, score_max = correct_ans(quiz, quiz_id, given, weights=weights)
            except Exception as e:
                print(f"Erreur pour la correction de {s}, quiz_id={quiz_id}")
                #print(given, type(given), e)
            Res.loc[s, quiz_id] = score/score_max
        Res.loc[s, 'maxpts'] = pd.Series([bareme[q] for q in exam_questions[s]]).sum()
    Res = Res.infer_objects(copy=False).dropna(axis=1, how="all").fillna(0)
    if seuil==0: Res[Res < 0] = 0
    poids = pd.Series({q: bareme[q] for q in Res.columns if q != 'maxpts'})
    Res['Note'] = Res[poids.index].dot(poids) / Res['maxpts'] * 20
      
    return Res

#correctAllBis(students_answers, quiz, data_filt, seuil=0, exam_questions=exam_questions, weights=None, bareme=None, maxtries=3)
# Res = correctAllBis(students_answers, quiz, data_filt, seuil=0, exam_questions=None, weights=None, bareme=None, maxtries=3)
"""
# All students case
import time
tic = time.perf_counter()
ResDataFiltBis = correctAllBis(students_answers, quiz, 
                            data_filt, bareme={ 'quiz13':0, 'quiz55':0, 'quiz56':0, 'quiz57':0})
toc = time.perf_counter()
print(f"Temps d'exécution : {toc-tic:.3f} seconde(s)")
"""

"""
#Random multiple exams
#quiz.exam_show(exam_title = "Examen de Truc",   shuffle=True, nb=2) # <-- the exam itself
exam_questions = getExamQuestions("Examen de Truc", data)
students = exam_questions.keys()
students_answers = getAllStudentsAnsvers(students, data, maxtries=3)
correctAllBis(students_answers, quiz, data_filt, seuil=0, exam_questions=exam_questions, weights=None, bareme=None, maxtries=3)
"""

"""
## 1 - Lecture des données
URL = "https://URL_UTILISÉE_POUR_RECUEILLIR_LES_RÉSULTATS"
SECRET = "MOT_DE_PASSE_SECRET_SPÉCIFIÉ_DANS_LE_SHEET"
QUIZFILE = "quizzes_basic_filtering_test.yml"
"""

def correctQuizzes(URL, SECRET, QUIZFILE, title=None, seuil=0, weights=None, bareme=None, maxtries=1):
    """
    Correct all quizzes in URL
    Arguments
    ---------    
        URL:         str
             Adresse du google sheet sur laquelle aller chercher les données
        SECRET:      str
             Code SECRET utilisé pour accéder aux données
        QUIZFILE:    str
             Nom du fichier de quiz non encodé CONTENANT les valeurs attendues 
        title:       str None par défaut
            Si title n'est pas None, c'est qu'on doit corriger un test avec tirage au sort des questions
            de type exam_show,  de titre title
        seuil:      float
            seuil=0 seuille à zéros les notes de chaque question (sinon note négative possible)
        weights:    dict
            Matrice de poids 
            ex: default_weights = {
                (True, True):   1,  # Vrai Positif
                (True, False): -1,  # Faux Positif 
                (False, True):  0,  # Faux Négatif (oubli)
                (False, False): 0   # Vrai Négatif
            }
        bareme:     dict
            Barême = poids des différentes questions dans le quiz. Si pas de barême, toutes les 
            questions sont au même poids pour le calcul de la note. Si poids d'une question 
            non spécifié, il est à 1 par défaut. ex: bareme = {'quiz3':4, 'quiz55':0} (tous les autres 
            avec un poids de 1).
        maxtries:   int
            Nombre d'essais permis
        
    Returns
    -------
        Res:       pandas dataframe
            lignes : étudiants (`students`), colonnes : chacun des quizzes + une colonne Note 
            
    """
    from labquiz import QuizLab
    from labquiz.putils import readData, getAllStudentsAnsvers, getExamQuestions, correctAll

    ## 1 - Lecture des données
    data, data_filt = readData(URL, SECRET)
    
    if title is None:
        ## 2 - identification des participants et extraction des réponses
        students = sorted(list(data_filt["student"].dropna().unique()))
        exam_questions = None
    else:
        exam_questions = getExamQuestions(title, data)
        students = exam_questions.keys()
    
    students_answers = getAllStudentsAnsvers(students, data, maxtries=maxtries) 

    ## 3 - Instancier un quiz avec le fichier de quiz CONTENANT les valeurs attendues
    URL = ""
    quiz = QuizLab(URL, QUIZFILE, needAuthentification=False, retries=0)

    ## 4 - Corriger !
    ResDataFilt = correctAll(students_answers, quiz, data_filt, seuil=seuil, 
                    exam_questions=exam_questions, weights=weights, bareme=bareme, maxtries=maxtries)
    
    if bareme is not None:
        colsToDrop = [quiz_id for quiz_id, val in bareme.items() if val==0 and quiz_id in ResDataFilt.columns]
        return ResDataFilt.drop(columns=colsToDrop)
    return ResDataFilt


def correctQuizzesDf(data, data_filt, quiz, title=None, seuil=0, weights=None, bareme=None, maxtries=1):
    """
    Correct all quizzes in URL
    Arguments
    ---------    
        data, data_filt: pandas dataframes
             Les deux tableaux issus de readData
        quiz:    str
             instance du quiz avec fichier de quiz non encodé CONTENANT les valeurs attendues 
        title:       str None par défaut
            Si title n'est pas None, c'est qu'on doit corriger un test avec tirage au sort des questions
            de type exam_show,  de titre title
        seuil:      float
            seuil=0 seuille à zéros les notes de chaque question (sinon note négative possible)
        weights:    dict
            Matrice de poids 
            ex: default_weights = {
                (True, True):   1,  # Vrai Positif
                (True, False): -1,  # Faux Positif 
                (False, True):  0,  # Faux Négatif (oubli)
                (False, False): 0   # Vrai Négatif
            }
        bareme:     dict
            Barême = poids des différentes questions dans le quiz. Si pas de barême, toutes les 
            questions sont au même poids pour le calcul de la note. Si poids d'une question 
            non spécifié, il est à 1 par défaut. ex: bareme = {'quiz3':4, 'quiz55':0} (tous les autres 
            avec un poids de 1).
        maxtries:   int
            Nombre d'essais permis
        
    Returns
    -------
        Res:       pandas dataframe
            lignes : étudiants (`students`), colonnes : chacun des quizzes + une colonne Note 
            
    """
    from labquiz import QuizLab
    from labquiz.putils import readData, getAllStudentsAnsvers, getExamQuestions, correctAll

    ## 2 - identification des participants et extraction des réponses    
    if title is None:
        students = sorted(list(data_filt["student"].dropna().unique()))
        exam_questions = None
    else:
        exam_questions = getExamQuestions(title, data)
        students = exam_questions.keys()
    
    students_answers = getAllStudentsAnsvers(students, data, maxtries=maxtries) 


    ## 4 - Corriger !
    ResDataFilt = correctAll(students_answers, quiz, data_filt, seuil=seuil, 
                    exam_questions=exam_questions, weights=weights, bareme=bareme, maxtries=maxtries)
    
    if bareme is not None:
        colsToDrop = [quiz_id for quiz_id, val in bareme.items() if val==0 and quiz_id in ResDataFilt.columns]
        return ResDataFilt.drop(columns=colsToDrop)
    return ResDataFilt


def getAllStudentsAnsvers(students, data, maxtries=3):
    students_answers = {student:{} for student in students }
    data_filt =  data.query("event_type in ['validate', 'validate_exam', 'correction'] ")
    #
    for student in students:
        quizzes_done = data_filt.query(f"student == '{student}' ")["quiz_title"].unique()
        for quiz_id in quizzes_done:
            if quiz_id in ["0", "integrity"]: continue
            given = check_ans(student, quiz_id, data_filt, maxtries=maxtries)
            students_answers[student][quiz_id] = given
    return students_answers   
# exemple: students_answers = getAllStudentsAnsvers(students, data, maxtries=3)


def getExamQuestions(exam_title, data):
    """
    Get from `data` the specific questions asked for each students in the random quiz `exam_title`
    """
    import json
    dq = data.query(f"quiz_title == '{exam_title}' and event_type== 'starting'")
    dqq = dq.drop_duplicates(
       subset=["student"],
     keep="last")
    exam_questions = {}
    for row in dqq.index:
        exam_questions[dqq.loc[row, 'student']] = dqq.loc[row, 'answers']
    return exam_questions
#example
#exam_questions = getExamQuestions("Examen de Truc", data)


def correctAllPrev(students_answers, quiz, df, seuil=0, weights=None, bareme=None, maxtries=3):
    """Corrige toutes les réponses
    
    correctAll(students, sols, df, seuil=0, Poids=None, bareme=None)
    return Res 
    Corrige l'ensemble des résultats des étudiants de `students` à partir des solutions `sols`
    
    Arguments
    ---------
        students:   dict
            Réponses données
        quiz:      objet 
            Objet QuizLab dans lequel a été chargé le sujet
            >> quiz = QuizLab(URL, QUIZFILE, needAuthentification=False, retries=100)
        df:         pandas dataframe
            Tableau d'entrée
        seuil:      float
            seuil=0 seuille à zéros les notes de chaque question (sinon note négative possible)
        weights:    dict
            Matrice de poids 
            ex: default_weights = {
                (True, True):   1,  # Vrai Positif
                (True, False): -1,  # Faux Positif 
                (False, True):  0,  # Faux Négatif (oubli)
                (False, False): 0   # Vrai Négatif
            }
        bareme:     dict
            Barême = poids des différentes questions dans le quiz. Si pas de barême, toutes les 
            questions sont au même poids pour le calcul de la note. Si poids d'une question 
            non spécifié, il est à 1 par défaut. ex: bareme = {'quiz3':4} (tous les autres 
            avec un poids de 1).
        maxtries:   int
            Nombre d'essais permis
        
    Returns
    -------
        Res:       pandas dataframe
            lignes : étudiants (`students`), colonnes : chacun des quizzes + une colonne Note 

    """
    import pandas as pd
    
    allquizzes = [q for q in quiz.quiz_bank.keys() if q != "title"]
    students = students_answers.keys()
    Res = pd.DataFrame(index=students, columns=allquizzes)
    if bareme is not None: 
        bareme = {q:1 if q not in bareme.keys() else bareme[q] for q in allquizzes }
    else:
        bareme = {q:1 for q in allquizzes}
    for s in students:
        for quiz_id in allquizzes:
            given = students_answers[s].get(quiz_id, {})
            if given == {}: continue
            try:
                score, score_max = correct_ans(quiz, quiz_id, given, weights=weights)
            except:
                print(f"Erreur pour la correction de {s}, quiz_id={quiz_id}")
            Res.loc[s, quiz_id] = score/score_max
    Res = Res.infer_objects(copy=False).fillna(0)
    if seuil==0: Res[Res < 0] = 0
    poids = pd.Series(bareme)
    Res['Note'] = Res[poids.index].dot(poids) / poids.sum()*20
      
    return Res
"""
students = sorted(list(data["student"].dropna().unique()))
students = [s.title().strip() for s in students]
students_answers = getAllStudentsAnsvers(students, data, maxtries=3)

import time
tic = time.perf_counter()
ResDataFiltBis = correctAllBis(students_answers, quiz, 
                            data_filt, bareme={ 'quiz13':0, 'quiz53':0, 'quiz54':0,
                                               'quiz55':0, 'quiz56':0, 'quiz57':0})
toc = time.perf_counter()
print(f"Temps d'exécution : {toc-tic:.3f} seconde(s)")
"""


#=====================================================
# Sécurité 
# --------
# Détecte si les paramètres de démarrage
# ou au cours de l'exécution ont été modifiés. Détecte
# si le source a été modifié ou monkey patched.
#=====================================================



def start_integrity(starting_values, df):
    """Vérifie que les paramètres de démarrage n'ont pas été modifiés 
       (le nom de l'étudiant n'est pas forcément connu à ce niveau)
    
    start_integrity(tarting_values, df)
    return bool
    
    
    Parameters
    ----------

        starting_values:    dict
            Dictionnaire des clés et valeurs à tester pour modification éventuelle
        df:     pandas dataframe
            Tableau de données
    Returns
    -------
        bool
    
    """
    # Vérifie que les paramètres de démarrage n'ont pas été modifiés
    #dfs = df.query(f"student == '{s}' ")
    allans = df.query(f"event_type == 'starting'")[['notebook_id','parameters','answers']]
    for idx in allans.index:
        params = allans.loc[idx, 'parameters']
        id = allans.loc[idx, 'notebook_id']
        #ans = parse_custom_dict(ans)
        subans = {key:params.get(key,'') for key in starting_values} #sous ensemble de ans avec les bonnes clés
        if not subans == starting_values:
            for key in starting_values:
                if subans[key] != starting_values[key]: 
                    print(f"{id} - enregistrement {idx} : Clé originale '{key}' modifiée de {starting_values[key]} vers {subans[key]}")
            #return False
    return True


        
def check_start_integrity_all_std(starting_values, df):
    """vérification d'intégrité pour l'ensemble des étudiants   
    
    check_start_integrity_all_std(starting_values, df)
    
    Parameters
    ----------
        starting_values:    dict
            Dictionnaire des clés et valeurs à tester pour modification éventuelle
        df:       pandas dataframe
            Tableau de données
    Returns
    -------
        nothing
    """

    start_integrity(starting_values, df)

########        
        

def check_integrity_msg(s, parameters, df):
    """Vérifie que les paramètres n'ont pas été modifiés au cours de l'exécution pour l'étudiant `s`
    
    check_integrity(s, parameters, df)
    return bool
    Pour l'étudiant `s`, vérifier que
      - machine_id ne change pas
      - les paramètres de fonctionnement (contenus dans check_integrity) n'ont pas été modifiés
    Affiche si modification de machine, paramètres modifiés (et lesquels)   
    
    Parameters
    ----------
        s:     str
            Student name
        parameters:    dict
            Dictionnaire des clés et valeurs à tester pour modification éventuelle
        df:     pandas dataframe
            Tableau de données
    Returns
    -------
        bool
    
    """
    out = True
    msg = []
    dfs = df.query(f"student == '{s}' ")
    # - machine_id ne change pas
    if not (len(dfs["notebook_id"].unique()) == 1):
        msg.append(f"{s}: Modification machine pour le même nom d'étudiant")
        msg.append(str(dfs["notebook_id"].unique()))
        out = False

    # vérifier que les paramètres check_integrity n'ont pas été modifiés
    recorded_param = dfs['parameters']
    if any( pd.isnull(recorded_param.loc[idx]) for idx in recorded_param.index):
        # c'est qu'ils n'ont pas été enregistrés dans parameters mais dans answers
        recorded_param = dfs.query(f"quiz_title == 'integrity' & event_type == 'check_integrity'")['answers']
    if len(recorded_param)==0: 
        return out,  "\n".join(msg) #Pas de check pour cet étudiant
    for idx in recorded_param.index: 
        #ans = parse_custom_dict(ans.values[0])
        subrecord = {key:recorded_param.loc[idx].get(key, '') for key in parameters}
        if not subrecord == parameters:
            msg.append(f"{s} - enregistrement {idx} :")
            for key in parameters:
                if subrecord[key] != parameters[key]:  
                    msg.append(f"      - Clé originale '{key}' modifiée de {parameters[key]} vers {subrecord[key]}")
            out = False  
    return out, "\n".join(msg)


def check_integrity(s, parameters, df):
    """Vérifie que les paramètres n'ont pas été modifiés au cours de l'exécution pour l'étudiant `s`
    
    check_integrity(s, parameters, df)
    return bool
    Pour l'étudiant `s`, vérifier que
      - machine_id ne change pas
      - les paramètres de fonctionnement (contenus dans check_integrity) n'ont pas été modifiés
    Affiche si modification de machine, paramètres modifiés (et lesquels)   
    
    Parameters
    ----------
        s:     str
            Student name
        parameters:    dict
            Dictionnaire des clés et valeurs à tester pour modification éventuelle
        df:     pandas dataframe
            Tableau de données
    Returns
    -------
        bool
    
    """
    out = True
    dfs = df.query(f"student == '{s}' ")
    # - machine_id ne change pas
    if not (len(dfs["notebook_id"].unique()) == 1):
        print(f"{s}: Modification machine pour le même nom d'étudiant")
        print(dfs["notebook_id"].unique())
        out = False

    # vérifier que les paramètres check_integrity n'ont pas été modifiés
    recorded_param = dfs['parameters']
    if any( pd.isnull(recorded_param.loc[idx]) for idx in recorded_param.index):
        # c'est qu'ils n'ont pas été enregistrés dans parameters mais dans answers
        recorded_param = dfs.query(f"quiz_title == 'integrity' & event_type == 'check_integrity'")['answers']
    if len(recorded_param)==0: 
        return True #Pas de check pour cet étudiant
    for idx in recorded_param.index: 
        #ans = parse_custom_dict(ans.values[0])
        subrecord = {key:recorded_param.loc[idx].get(key, '') for key in parameters}
        if not subrecord == parameters:
            print(f"{s} - enregistrement {idx} :")
            for key in parameters:
                if subrecord[key] != parameters[key]:  
                    print(f"      - Clé originale '{key}' modifiée de {parameters[key]} vers {subrecord[key]}")
            out = False  
    return out

def check_integrity_all_std(parameters, students, df):  
    """vérification d'intégrité pour l'ensemble des étudiants dans students   
    
    check_start_integrity_all_std(starting_values, students, df)
    
    Parameters
    ----------
        starting_values:    dict
            Dictionnaire des clés et valeurs à tester pour modification éventuelle
        students: list
            Liste de str (noms des étudiants)
        df:       pandas dataframe
            Tableau de données
    Returns
    -------
        nothing
    """
    for s in students:
        check_integrity(s, parameters, df)

def check_machine(df):
    """Détecte si une même machine a été utilisée pour plusieurs noms d'étudiants
    
    check_machine(df)
    
    Parameters
    ----------
        df:       pandas dataframe
            Tableau de données
    Returns:
    -------
        nothing (Alert for cases)
    """
    
    notebookids = df["notebook_id"].unique()
    for nb in notebookids:
        dfs = df.query(f"notebook_id == '{nb}' ")             
        if not (len(dfs["student"].unique()) == 1):
            print(f"Même machine {nb} utilisée par plusieurs étudiants")
            print(dfs['student'].unique())

def machines_for_std(s, df):
    """retourne numéro(s) de machines pour un étudiant `s`
    
    machines_for_std(s, df)
    
    Parameters
    ----------
        s:        str
            Nom de l'étudiant
        df:       pandas dataframe
            Tableau de données
    Returns:
    -------
        list of machines_id for `s`

    """
    dfs = df.query(f"student == '{s}' ")
    return dfs['notebook_id'].unique()       
    
def stds_for_machine(id, df):
    """retourne les noms d'étudiant(e)(s) pour un numéro de machine
    
    stds_for_machine(id, df)
    
    Parameters
    ----------
        id:        str
            Identifiant de machine
        df:       pandas dataframe
            Tableau de données
    Returns:
    -------
        list of students having used machines_id

    """
    # noms d'étudiant(e)(s) pour un numéro de machine
    dfs = df.query(f"notebook_id == '{id}' ")
    return dfs['student'].unique()

def check_hash_integrity(df, kind_hash='full', wanted_hash=""):
    
    """test d'intégrité (modification du source, monkey patching, paramètres surveillés)
    
    check_hash_integrity(data, kind_hash='full', wanted_hash="")
    return: nothing
    
    Parameters
    ----------
        kind_hash:        str
            Type de hash (`src`ou `full`) 
            src pour les sources, indépendant machine et `full` dépend des src et paramètres
        df:       pandas dataframe
            Tableau de données
        wanted_hash: str
            Si test du hash source, la référence est `wanted_hash`
    Returns:
    -------
        nothing (Alert for cases)

    """
    
    # test kind_hash: src ou full
    #dfnew = df.query(f"quiz_title == 'integrity' & event_type == 'check_integrity'")[['timestamp', 'notebook_id', 'student', 'parameters', 'answers']]
    dfnew = df[['timestamp', 'notebook_id', 'student', 'parameters', 'answers']].copy()
    
    def get_hash(row, kind_hash):
        params = row.get("parameters")
        if isinstance(params, dict) and kind_hash in params:
            return params[kind_hash]
        answers = row.get("answers") #fallback car avant on rangeait le hash dans answers
        if isinstance(answers, dict):
            return answers.get(kind_hash, "")
        return ""
  
    dfnew.loc[:,"src_hash"] = dfnew.apply(lambda row: get_hash(row, 'src_hash'), axis=1)
    dfnew.loc[:,"full_hash"] = dfnew.apply(lambda row: get_hash(row, 'full_hash'), axis=1)
    
    ids = dfnew.groupby(['student', "notebook_id"])
    colhash='full_hash' if (kind_hash == 'full') else 'src_hash'
    for n, group in enumerate(ids):
        student = group[0][0]
        machine_id = group[0][1]
        g = group[1][colhash]
        if (len(g.unique()) > 1): 
            s = []
            for h in g.unique():
                s.append(dfnew.query(f"{colhash} == '{h}'")['student'].iloc[-1])
              
            if kind_hash=='src': print(f"Le source a été modifié pour {s}, machine id {id}")
            if kind_hash=='full': 
                print(f"⚠️ {student}, machine id {machine_id} :")
                print(f"    👉🏼 Le source ou les paramètres ont été modifiés ou monkey patched")
            print("hash constatés :")
            
            idx = g[g != g.shift()]
            for n,h in enumerate(g.unique()):
                msg = f"⚠️ index [{idx.index[n]}]" if h != wanted_hash else "👍"
                print(h, msg)
            print()
        else:
            if (g.unique() != wanted_hash):
                print(f"⚠️ {student}, machine id {machine_id} :")
                print(f"    👉🏼 Le source ou les paramètres ont été modifiés ou monkey patched")
                print(f"⚠️ index {g.index[0]} hash: {g.loc[g.index[0]]}")

def check_hash_integrity_msg(df, kind_hash='full', wanted_hash=""):
    
    """test d'intégrité (modification du source, monkey patching, paramètres surveillés)
    
    check_hash_integrity_msg(data, kind_hash='full', wanted_hash="")
    return: nothing
    
    Parameters
    ----------
        kind_hash:        str
            Type de hash (`src`ou `full`) 
            src pour les sources, indépendant machine et `full` dépend des src et paramètres
        df:       pandas dataframe
            Tableau de données
        wanted_hash: str
            Si test du hash source, la référence est `wanted_hash`
    Returns:
    -------
        nothing (Alert for cases)

    """
    
    # test kind_hash: src ou full
    #dfnew = df.query(f"quiz_title == 'integrity' & event_type == 'check_integrity'")[['timestamp', 'notebook_id', 'student', 'parameters', 'answers']]
    dfnew = df[['timestamp', 'notebook_id', 'student', 'parameters', 'answers']].copy()
    
    msg = []
    
    def get_hash(row, kind_hash):
        params = row.get("parameters")
        if isinstance(params, dict) and kind_hash in params:
            return params[kind_hash]
        answers = row.get("answers") #fallback car avant on rangeait le hash dans answers
        if isinstance(answers, dict):
            return answers.get(kind_hash, "")
        return ""
  
    dfnew.loc[:,"src_hash"] = dfnew.apply(lambda row: get_hash(row, 'src_hash'), axis=1)
    dfnew.loc[:,"full_hash"] = dfnew.apply(lambda row: get_hash(row, 'full_hash'), axis=1)
    
    ids = dfnew.groupby(['student', "notebook_id"])
    colhash='full_hash' if (kind_hash == 'full') else 'src_hash'
    for n, group in enumerate(ids):
        student = group[0][0]
        machine_id = group[0][1]
        g = group[1][colhash]
        if (len(g.unique()) > 1): 
            s = []
            for h in g.unique():
                s.append(dfnew.query(f"{colhash} == '{h}'")['student'].iloc[-1])
              
            if kind_hash=='src': print(f"Le source a été modifié pour {s}, machine id {id}")
            if kind_hash=='full': 
                msg.append(f"⚠️ {student}, machine id {machine_id} :")
                msg.append(f"    👉🏼 Le source ou les paramètres ont été modifiés ou monkey patched")
            print("hash constatés :")
            
            idx = g[g != g.shift()]
            for n,h in enumerate(g.unique()):
                msg.append(f"{h}: ⚠️ index [{idx.index[n]}]" if h != wanted_hash else "👍")
        else:
            if (g.unique() != wanted_hash):
                msg.append(f"⚠️ {student}, machine id {machine_id} :")
                msg.append(f"    👉🏼 Le source ou les paramètres ont été modifiés ou monkey patched")
                msg.append(f"⚠️ index {g.index[0]} hash: {g.loc[g.index[0]]}")
                
    return '\n'.join(msg)
                
#########

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def fernet_key_from_password(password: str) -> bytes:
    return base64.urlsafe_b64encode(
        hashlib.sha256(password.encode()).digest()
    )

def session_sentinel(pw2: str) -> str:
    return hashlib.sha256(pw2.encode()).hexdigest()


def inject_sentinel(py_path: Path, sentinel: str):
    
    _SENTINEL_RE = re.compile(
    r"^\s*#\s*noqa:\s*E501\s*#\s*.*$",
    re.MULTILINE,
    )
    
    text = py_path.read_text(encoding="utf-8")
    #line = f"# __SESSION_SENTINEL__ = {sentinel}"
    line = f"# noqa: E501  # {sentinel[:16]}"

    if _SENTINEL_RE.search(text):
        text = _SENTINEL_RE.sub(line, text)
    else:
        text = text.rstrip() + "\n" + line + "\n"

    py_path.write_text(text, encoding="utf-8")

    
def hash_distribution(files: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(files):
        h.update(p.read_bytes())
    return h.hexdigest()


def create_secure_tar(
    source_dir,
    output_file,
    password_open,     # PW₁
    password_seal      # PW₂ (secret)
):
    #import io, tarfile
    
    source_dir = Path(source_dir)
    output_file = Path(output_file)
    

    # --- Insertion d'une sentinelle dans les fichiers sources ---
    
    SENTINEL = session_sentinel(password_seal)

    for py in source_dir.rglob("*.py"):
        inject_sentinel(py, SENTINEL)

        
    # --- création du tar  ---
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add(source_dir, arcname=".")
    tar_bytes = buf.getvalue()

    global_hash = sha256_bytes(tar_bytes) # --- hash du tar ---

    # --- chiffrement ---
    key = fernet_key_from_password(password_open)
    encrypted = Fernet(key).encrypt(tar_bytes)
    output_file.write_bytes(encrypted)

    # 
    src_hash = hash_distribution(list(source_dir.rglob("*.py")))

    print("✔ Archive créée :", output_file)
    print("🔐 Hash global intégrité de l'archive (à conserver) :")
    print(global_hash)
    print("🔐 Hash global des fichiers sources et données (à conserver) : ")
    print(src_hash)

    return global_hash, src_hash

### Check integrity en cours d'exam 
# Préparation de l'archive 
# Exemple : 
"""global_hash, src_hash = create_secure_tar(
    source_dir="quiz_data",
    output_file="quiz.tar.enc",
    password_open="mot_pour_ouvrir",
    password_seal="secret_enseignant22"
)"""
#
# ----- Difference between dicts (recursive) -------------
def diff_dicts(
    row_idx,
    current,
    reference,
    ignore_keys=None,
    ignore_paths=None,
    path=""
):
    # Return:
    # anomaly: bool
    # anomalies : str - message describing found anomalies
    # out: dict - dict of anomalies {key:(reference_value, new_value) ...}
    
    ignore_keys = ignore_keys or set()
    ignore_paths = ignore_paths or set()
    anomaly = False
    anomalies = []
    out = {}

    all_keys = set(reference) | set(reference)

    for key in all_keys:
        if key in ignore_keys:
            continue

        cur_path = f"{path}.{key}" if path else key
        if cur_path in ignore_paths:
            continue

        v_cur = current.get(key, "<absente>")
        v_ref = reference.get(key, "<absente>")

        # Cas dictionnaires imbriqués
        if isinstance(v_cur, dict) and isinstance(v_ref, dict):
            anomalies.extend(
                diff_dicts(
                    row_idx,
                    v_cur,
                    v_ref,
                    ignore_keys,
                    ignore_paths,
                    cur_path
                )
            )
        else:
            if v_cur != v_ref:
                anomalies.append(
                    f"Ligne {row_idx} - Anomalie, pour la clé '{cur_path}', "
                    f"valeur de départ {v_ref} modifiée en {v_cur}"
                )
                anomaly = True
                out[cur_path] = (v_ref, v_cur)

    return anomaly, anomalies, out

def make_ano_report(out, includeRAS=True):
    if len(out) > 0:
        text = "⚠️" 
    else: 
        text = "✅ RAS"
        return text
    for k,v in out.items():
        if k=='full_hash':
            v0, v1 = str(v[0]), str(v[1])
            hash1 = v0[0:5] + '...' + v0[-3:]
            hash2 = v1[0:5] + '...' + v1[-3:]
            text += f"\n key {k} modified from {hash1} to {hash2}"
        else:
            text += f"\n key {k} modified from {v[0]} to {v[1]}"
    return text

def make_anomalies_df_report(df, reference, ignore_keys=[], includeRAS=True):
    # Generate the dataframe report
    messages = []
    Out = {}

    for idx, params in sorted(df["parameters"].items()):
        if isinstance(params, dict) and len(params) != 0:
            dd = diff_dicts(
                    row_idx=idx,
                    current=params,
                    reference=reference,
                    ignore_keys=ignore_keys,
                    ignore_paths=[]
                )
            if dd[0]:
                messages.extend(dd[1])
                Out[idx] = dd[2]
            else:
                if includeRAS:
                    Out[idx] = '' 
                else:
                    pass
                #print(f"pas de souci pour {idx}")
        else:
            pass
            #print(f'idx {idx} pas de dictionnaire')

    Result = pd.DataFrame(columns=['idx', 'timestamp', 'student', 'anomalies'])
    for idx, out in Out.items():
        if not pd.isnull(df.loc[idx, 'student']):
            row = idx, df.loc[idx, 'timestamp'], df.loc[idx, 'student'], make_ano_report(out, includeRAS)
            Result.loc[len(Result)] = row

    Result['timestamp'] = pd.to_datetime(Result["timestamp"].str.replace(r"\s*\(.*\)$", "", regex=True), errors="coerce" )
    Result['timestamp'] = Result['timestamp'].dt.strftime("%Y-%m-%d %H:%M")
    
    return Result

def group_anomalies_per_student(Result):
    df2 = Result.copy()
    if not df2.empty:
        indent = ' '*len(df2.loc[df2.index[0], "timestamp"] + ":")
    else: 
        indent = ""
    df2["entry"] = df2["timestamp"] + ":" + \
df2["anomalies"].str.replace('\n', f"\n {indent}- ")
    #.str.strip('\n').str.replace('\n', f"\n {indent}")

    Result_grouped = (
        df2.sort_values("timestamp")
           .groupby("student", as_index=False)
           .agg(texte=("entry", "\n".join))
    )
    return Result_grouped
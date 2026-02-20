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
    # Reading the YAML file
    with open(input_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Shuffle the order of propositions for each quiz
    for quiz_name, quiz_content in data.items():
        if "propositions" in quiz_content:
            random.shuffle(quiz_content["propositions"])

    # Saving in a new file
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
    # Input: initial or mixed question file 
    # # Output: File without solutions, answers or tips 
    # # Reading the YAML file
    with open(input_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # removing hints for each quiz
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
            
    # Saving in a new file
    with open(output_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False
        )


def quiz_to_dict_of_solutions(input_file):
    """
    extracts the solutions from the initial file

    Settings
    ----------
        input_file:str
            original or mixed question file    
    Returns
    -------
        out: dict
            dictionary {quiz1: {label1:expected, label2:expected, ...}, 
            quiz2:{label1:expected, label2:expected, ...} }
    """
    # Reading the YAML file
    with open(input_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 
    dico = {quiz_name:  {prop["label"]:prop["expected"] for prop in quiz_content["propositions"]} 
            for quiz_name, quiz_content in data.items()}
    out = {l: dict(sorted(d.items(), key=lambda item: item[0]))
                       for l,d in zip(dico.keys(), dico.values())}
    # We sort on the keys in case of!
    
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

    # Shuffle the order of propositions in each quiz
    for quiz_name, quiz_content in data.items():
        if "propositions" in quiz_content:
            random.shuffle(quiz_content["propositions"])
            
    # Questions only
    data_only = copy.deepcopy(data)
    # removing hints for every quiz
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
    """Reading data

    readData(URL, SECRET)
    return df, df_last
    Connects to retrieve the data table designated by the URL address and protected by SECRET

    Settings
    ----------
        URL: str
                Path
        SECRET: str
                String used to access data
    Returns
    -------
        df: pandas dataframe
                complete table
        df_filt:pandas dataframe
                table filtered just on validations and corrections
    """
    
    #global df, df_last

    try:
        r = requests.get(URL, params={"secret": SECRET})
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text)) 
        df.student = df.student.apply(lambda s: s.strip() if isinstance(s, str) else s)  #Normalization of student column
        df["answers"] = df.answers.apply(lambda x: parse_custom_dict(x)) #Decoding answers column
        df["parameters"] = df.parameters.apply(lambda x: parse_custom_dict(x)) #Decoding answers column
        now = datetime.now().strftime("%H:%M:%S")

        footer_msg = f"<span style='color:green; font-weight:bold'>✔ Sheet loaded ... ({len(df)} rows) - {now} </span>"
        display(HTML(footer_msg))

    except Exception as e:
        footer_msg =  f"<span style='color:red; font-weight:bold'>✘ Loading error {e}</span>"
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
    """Conversion of answers (str) to dictionary

    convert_ans(answers)
    return dictionary
    Converts responses (string) to response dictionary

    Settings
    ----------
        answers: str
    Returns
    -------
        dico: dict
                    matching dictionary
    """

    try:
        dico = {k: v.lower() == "true" for k, v in (item.rsplit("=",1) for item in answers.strip("{}").split(", "))}
        dico = dict(sorted(dico.items(), key=lambda item: item[0])) # We sort keys, never mind
    except:
        print(answers)
    return dico


def correct_ans(quiz, quiz_id, given, weights=None, constraints=None):
    """
    Correction of an answer
    # correct_ans(quiz, quiz_id, given, weights=None)
    # return score, score_max
    Correct the answers given in `given` by comparing with those contained
    in the subject (with correction) for the label `quiz_id`. 

    Settings
    ----------
        quiz: object 
            QuizLab object into which the topic was loaded
            >> quiz = QuizLab(URL, QUIZFILE, needAuthentification=False, retries=100)
        quiz_id: str
            Key to the desired quiz
        given: dict 
            Answers given
        weights: pandas dataframe
            Weight matrix 
            ex: default_weights = {
                (True, True): 1, # True Positive
                (True, False): -1, # False Positive 
                (False, True): 0, # False Negative (forget)
                (False, False): 0 # True Negative
                }
    Returns
    -------
        score:float
                score obtained
        score_max: float
                maximum score that can be achieved
    """

    from .utils import calculate_quiz_score
    question, quiz_type, propositions, constraints = quiz._QuizLab__load_quiz(quiz_id)
    #NB: The name _QuizLab__load_quiz corresponds to the dunder __loadquiz method in main
    propositions.sort(key=lambda d: d["label"]) # tri sur les clés
    given = {k:given[k] for k in sorted(given)}
    #print(given, propositions)
    return calculate_quiz_score(quiz_type, given, propositions, weights=weights, constraints=constraints)


def check_ans(student, quiz_id, df, maxtries=3):
    """Selects the response to use

    check_ans(student, quiz_id, df, maxtries=maxtries)
    return given
    Select the “correct” answer: 
    - that of rank <= maxtries 
    - if correction requested: the last before correction if <= maxtries otherwise maxtries 

    Arguments
    ---------
        student:str
            student name to search in the table
        quiz_id: str
            id of the string to use (to search in the table)
        df: pandas dataframe
            data table
        maxtries: int
            Max number of attempts
    Returns
    -------
        given: dict
            answer dictionary
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



def new_parse_custom_dict(text):  # Format change!! Much simpler
    """Converts the character string answers into a dictionary

    parse_custom_dict(text)
    return result
    With format change, just json decoding

    Settings
    ----------
        text:str
            character string in the format '{"val1":true, "val2":false, "val3":float, "val4":"{...}"}'
    Returns
    -------
        result: dict
            sorted dictionary of identified keys:values       
    """

    import json
    d = json.loads(text)
    if isinstance(d, dict):
        d_trie = {k: v for k, v in sorted(d.items())}
        return d_trie
    return d

def old_parse_custom_dict(text):
    """Converts a string into a dictionary

    parse_custom_dict(text)
    return result
    Converts a character string to the format "{val1=true, val2=false, val3=float, val4={...}}"
    to a dictionary containing boolean values, numbers, or other dictionaries

    Settings
    ----------
        text:str
            character string in the format "{val1:true, val2=false, val3=float, val4={...}}"
    Returns
    -------
        result: dict
            dictionary of keys:identified values

    #Thanks to Gemini!!         
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
    """Converts string of answers in df['answers'] to dictionary, keys sorted
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
    """Correct all answers

    correctAll(students, floors, df, threshold=0, Weight=None, scale=None)
    return Res 
    Corrects all student results from `students` using `sols` solutions

    Arguments
    ---------
        students: dict
            Answers given
        quiz: object 
            QuizLab object into which the topic was loaded
            >> quiz = QuizLab(URL, QUIZFILE, needAuthentification=False, retries=100)
        df: pandas dataframe
            Input table
        threshold: float
            threshold=0 thresholds the marks for each question to zero (otherwise negative mark possible)
        exam_questions: list (default None)
            list of questions to correct (by default all)
        weights: dict
            Weight matrix 
            ex: default_weights = {
                (True, True): 1, # True Positive
                (True, False): -1, # False Positive 
                (False, True): 0, # False Negative (forget)
                (False, False): 0 # True Negative
            }
        scale: dict
            Scale = weight of the different questions in the quiz. If no scale, all 
            questions have the same weight for calculating the grade. If weight of a question 
            not specified, it is 1 by default. ex: scale = {'quiz3':4, 'quiz55':0} (all others 
            with a weight of 1).
        maxtries: int
            Number of attempts allowed
        
    Returns
    -------
        Res: pandas dataframe
            rows: students, columns: each quiz + a Note column 

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
            #print(f"Correction de {s}, quiz_id={quiz_id}")
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
        URL:str
             Address of the Google sheet on which to fetch the data
        SECRET: str
             SECRET code used to access data
        QUIZFILE:str
             Unencoded quiz file name CONTAINING expected values 
        title: str None by default
            If title is not None, it is because we must correct a test with random selection of questions
            of type exam_show, of title title
        threshold: float
            threshold=0 thresholds the marks for each question to zero (otherwise negative mark possible)
        weights: dict
            Weight matrix 
            ex: default_weights = {
                (True, True): 1, # True Positive
                (True, False): -1, # False Positive 
                (False, True): 0, # False Negative (forget)
                (False, False): 0 # True Negative
            }
        scale: dict
            Scale = weight of the different questions in the quiz. If no scale, all 
            questions have the same weight for calculating the grade. If weight of a question 
            not specified, it is 1 by default. ex: scale = {'quiz3':4, 'quiz55':0} (all others 
            with a weight of 1).
        maxtries: int
            Number of attempts allowed
        
    Returns
    -------
        Res: pandas dataframe
            rows: students, columns: each quiz + a Note column 
            
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
             The two tables from readData
        quiz: str
             quiz instance with non-encoded quiz file CONTAINING expected values 
        title: str None by default
            If title is not None, it is because we must correct a test with random selection of questions
            of type exam_show, of title title
        threshold: float
            threshold=0 thresholds the marks for each question to zero (otherwise negative mark possible)
        weights: dict
            Weight matrix 
            ex: default_weights = {
                (True, True): 1, # True Positive
                (True, False): -1, # False Positive 
                (False, True): 0, # False Negative (forget)
                (False, False): 0 # True Negative
            }
        scale: dict
            Scale = weight of the different questions in the quiz. If no scale, all 
            questions have the same weight for calculating the grade. If weight of a question 
            not specified, it is 1 by default. ex: scale = {'quiz3':4, 'quiz55':0} (all others 
            with a weight of 1).
        maxtries: int
            Number of attempts allowed
        
    Returns
    -------
        Res: pandas dataframe
            rows: students, columns: each quiz + a Note column 
            
    """
    from labquiz import QuizLab
    from labquiz.putils import readData, getAllStudentsAnsvers, getExamQuestions, correctAll

    ## 2 - identification of participants and extraction of responses   
    if title is None:
        students = sorted(list(data_filt["student"].dropna().unique()))
        exam_questions = None
    else:
        exam_questions = getExamQuestions(title, data)
        students = exam_questions.keys()
    
    students_answers = getAllStudentsAnsvers(students, data, maxtries=maxtries) 


    ## 4 - Correct all!
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
# example: students_answers = getAllStudentsAnsvers(students, data, maxtries=3)


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
    """Correct all answers
    correctAll(students, floors, df, threshold=0, Weight=None, scale=None)
    return Res 
    Corrects all student results from `students` using `sols` solutions

    Arguments
    ---------
        students: dict
            Answers given
        quiz: object 
            QuizLab object into which the topic was loaded
            >> quiz = QuizLab(URL, QUIZFILE, needAuthentification=False, retries=100)
        df: pandas dataframe
            Input table
        threshold: float
            threshold=0 thresholds the marks for each question to zero (otherwise negative mark possible)
        weights: dict
            Weight matrix 
            ex: default_weights = {
                (True, True): 1, # True Positive
                (True, False): -1, # False Positive 
                (False, True): 0, # False Negative (forget)
                (False, False): 0 # True Negative
            }
        scale: dict
            Scale = weight of the different questions in the quiz. If no scale, all 
            questions have the same weight for calculating the grade. If weight of a question 
            not specified, it is 1 by default. ex: scale = {'quiz3':4} (all others 
            with a weight of 1).
        maxtries: int
            Number of attempts allowed
        
    Returns
    -------
        Res: pandas dataframe
            rows: students, columns: each quiz + a Note column 

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
    Res['FinalMark'] = Res[poids.index].dot(poids) / poids.sum()*20
      
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
# Security
#
# --------
# Detects if startup parameters
# or during execution have been modified. Detects
# if the source has been modified or monkey patched.
#
#=====================================================



def start_integrity(starting_values, df):
    """Checks that startup settings have not been changed (the name of the student is not necessarily known at this level)

    start_integrity(tarting_values, df)
    return bool


    Settings
    ----------

        starting_values: dict
            Dictionary of keys and values to test for possible modification
        df: pandas dataframe
            Data table
    Returns
    -------
        bool

    """
    # Checks that startup parameters have not been changed
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
    """integrity check for all students

    check_start_integrity_all_std(starting_values, df)

    Settings
    ----------
        starting_values: dict
            Dictionary of keys and values to test for possible modification
        df: pandas dataframe
            Data table
    Returns
    -------
        nothing
    """

    start_integrity(starting_values, df)

########        
        

def check_integrity_msg(s, parameters, df):
    """Checks that settings were not changed during execution for student s

    check_integrity(s, parameters, df)
    return bool
    For student `s`, check that
    - machine_id does not change
    - the operating parameters (contained in check_integrity) have not been modified
    Shows if machine modification, parameters modified (and which ones)   

    Settings
    ----------
        s:str
            Student name
        parameters: dict
            Dictionary of keys and values to test for possible modification
        df: pandas dataframe
            Data table
    Returns
    -------
        bool

    """
    out = True
    msg = []
    dfs = df.query(f"student == '{s}' ")
    # - machine_id ne change pas
    if not (len(dfs["notebook_id"].unique()) == 1):
        msg.append(f"{s}: Machine modification for the same student name")
        msg.append(str(dfs["notebook_id"].unique()))
        out = False

    # verify that the check_integrity parameters have not been modified
    recorded_param = dfs['parameters']
    if any( pd.isnull(recorded_param.loc[idx]) for idx in recorded_param.index):
        # t is that they were not saved in parameters but in answers
        recorded_param = dfs.query(f"quiz_title == 'integrity' & event_type == 'check_integrity'")['answers']
    if len(recorded_param)==0: 
        return out,  "\n".join(msg) #No check for this student
    for idx in recorded_param.index: 
        #ans = parse_custom_dict(ans.values[0])
        subrecord = {key:recorded_param.loc[idx].get(key, '') for key in parameters}
        if not subrecord == parameters:
            msg.append(f"{s} - record {idx} :")
            for key in parameters:
                if subrecord[key] != parameters[key]:  
                    msg.append(f"      - Original key '{key}' modified from {parameters[key]} to {subrecord[key]}")
            out = False  
    return out, "\n".join(msg)


def check_integrity(s, parameters, df):
    """Checks that settings were not changed during execution for student s

    check_integrity(s, parameters, df)
    return bool
    For student `s`, check that
    - machine_id does not change
    - the operating parameters (contained in check_integrity) have not been modified
    Shows if machine modification, parameters modified (and which ones)   

    Settings
    ----------
        s:str
            Student name
        parameters: dict
            Dictionary of keys and values to test for possible modification
        df: pandas dataframe
            Data table
    Returns
    -------
        bool 
    """
    out = True
    dfs = df.query(f"student == '{s}' ")
    # - machine_id does not change
    if not (len(dfs["notebook_id"].unique()) == 1):
        print(f"{s}: Machine modification for the same student name")
        print(dfs["notebook_id"].unique())
        out = False

    # verify that the check_integrity parameters have not been modified
    recorded_param = dfs['parameters']
    if any( pd.isnull(recorded_param.loc[idx]) for idx in recorded_param.index):
        # It is that they were not saved in parameters but in answers
        recorded_param = dfs.query(f"quiz_title == 'integrity' & event_type == 'check_integrity'")['answers']
    if len(recorded_param)==0: 
        return True #No check for this student
    for idx in recorded_param.index: 
        #ans = parse_custom_dict(ans.values[0])
        subrecord = {key:recorded_param.loc[idx].get(key, '') for key in parameters}
        if not subrecord == parameters:
            print(f"{s} - enregistrement {idx} :")
            for key in parameters:
                if subrecord[key] != parameters[key]:  
                    print(f"      - Original key '{key}' modified from {parameters[key]} to {subrecord[key]}")
            out = False  
    return out

def check_integrity_all_std(parameters, students, df):  
    """
    Integrity check for all students in students

    check_start_integrity_all_std(starting_values, students, df)

    Settings
    ----------
        starting_values: dict
            Dictionary of keys and values to test for possible modification
        students: list
            List of str (student names)
        df: pandas dataframe
            Data table
    Returns
    -------
        nothing
    """
    for s in students:
        check_integrity(s, parameters, df)

def check_machine(df):
    """Detects if the same machine has been used for multiple student names
    
    check_machine(df)
    
    Parameters
    ----------
        df:       pandas dataframe
    Returns:
    -------
        nothing (Alert for cases)
    """
    
    notebookids = df["notebook_id"].unique()
    for nb in notebookids:
        dfs = df.query(f"notebook_id == '{nb}' ")             
        if not (len(dfs["student"].unique()) == 1):
            print(f"Same machine {nb} used by several students")
            print(dfs['student'].unique())

def machines_for_std(s, df):
    """Returns machine number(s) for a student s

    machines_for_std(s, df)

    Parameters
    ----------
        s:str
            Student Name
        df: pandas dataframe
            Data table
    Returns:
    -------
        list of machines_id for `s`

    """
    dfs = df.query(f"student == '{s}' ")
    return dfs['notebook_id'].unique()       
    
def stds_for_machine(id, df):
    """Returns student names for a machine number

    stds_for_machine(id, df)

    Parameters
    ----------
        id: str
            Machine ID
        df: pandas dataframe
            Data table
    Returns:
    -------
        list of students having used machines_id

    """
    dfs = df.query(f"notebook_id == '{id}' ")
    return dfs['student'].unique()

def check_hash_integrity(df, kind_hash='full', wanted_hash=""):
    
    """Integrity test (source modification, monkey patching, monitored parameters)

    check_hash_integrity(data, kind_hash='full', wanted_hash="")
    return: nothing

    Parameters 
    ----------
        kind_hash: str
            Hash type (`src` or `full`) 
            src for sources, machine independent and `full` depends on src and parameters
        df: pandas dataframe
            Data table
        wanted_hash: str
            If testing the source hash, the reference is `wanted_hash`
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
        answers = row.get("answers") #fallback because before the hash was stored in answers 
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
              
            if kind_hash=='src': print(f"Source was changed for {s}, machine id {id}")
            if kind_hash=='full': 
                print(f"⚠️ {student}, machine id {machine_id}:")
                print(f"    👉🏼 Source or settings have been changed or monkey patched")
            print("Observed hashes:")
            
            idx = g[g != g.shift()]
            for n,h in enumerate(g.unique()):
                msg = f"⚠️ index [{idx.index[n]}]" if h != wanted_hash else "👍"
                print(h, msg)
            print()
        else:
            if (g.unique() != wanted_hash):
                print(f"⚠️ {student}, machine id {machine_id} :")
                print(f"    👉🏼 Source or settings have been changed or monkey patched")
                print(f"⚠️ index {g.index[0]} hash: {g.loc[g.index[0]]}")

def check_hash_integrity_msg(df, kind_hash='full', wanted_hash=""):
    
    """Integrity test (source modification, monkey patching, monitored parameters)

    check_hash_integrity_msg(data, kind_hash='full', wanted_hash="")
    return: nothing

    Settings
    ----------
        kind_hash: str
            Hash type (`src` or `full`) 
            src for sources, machine independent and `full` depends on src and parameters
        df: pandas dataframe
            Data table
        wanted_hash: str
            If testing the source hash, the reference is `wanted_hash`
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
              
            if kind_hash=='src': print(f"Source code changed for {s}, machine id {id}")
            if kind_hash=='full': 
                msg.append(f"⚠️ {student}, machine id {machine_id} :")
                msg.append(f"    👉🏼 Source or settings have been changed or monkey patched")
            print("hash constatés :")
            
            idx = g[g != g.shift()]
            for n,h in enumerate(g.unique()):
                msg.append(f"{h}: ⚠️ index [{idx.index[n]}]" if h != wanted_hash else "👍")
        else:
            if (g.unique() != wanted_hash):
                msg.append(f"⚠️ {student}, machine id {machine_id} :")
                msg.append(f"    👉🏼 Source or settings have been changed or monkey patched")
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
    

    # --- Inserting a sentinel into source files ---
    
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

    print("✔ Archive created:", output_file)
    print("🔐 Global archive integrity hash (to keep):")
    print(global_hash)
    print("🔐 Global hash of source and data files (to be kept) ")
    print(src_hash)

    return global_hash, src_hash

### Check integrity during exam 
# Archive preparation
# Example : 
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

        # Nested dictionaries cases
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
                    f"Ligne {row_idx} - Anomaly, for key '{cur_path}', "
                    f"Initial value {v_ref} modified into {v_cur}"
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
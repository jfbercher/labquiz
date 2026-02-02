import ipywidgets as widgets
from IPython.display import display, update_display, Markdown, Javascript
import yaml, random, datetime, requests, json
import threading, time, sys, io
from io import BytesIO
from cryptography.fernet import Fernet
import asyncio
import random
from pathlib import Path
from .utils import get_full_object_hash

from cryptography.fernet import Fernet as Qwsp
Qwsp.weight = Qwsp.decrypt


# Patch pour request de manière à ne pas bloquer lors des logs
# Requête synchrone --> passe sur un thread

IS_JUPYTERLITE = "pyodide" in sys.modules or "piplite" in sys.modules
#TRUC = "test"

def patch_requests_post(callback=None):
    """
    Remplace requests.post par une version threadée (Jupyter classique uniquement).
    Le callback (optionnel) reçoit la réponse en argument.
    """
    if IS_JUPYTERLITE:
        print("JupyterLite détecté : Patch thread ignoré (non supporté).")
        return

    _original_post = requests.post

    def threaded_post(*args, **kwargs):
        def worker():
            try:
                response = _original_post(*args, **kwargs)
                if callback:
                    callback(response)
            except Exception as e:
                if callback:
                    callback(e)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return None # La fonction rend la main immédiatement au Kernel

    requests.post = threaded_post
    #print("Patch requests.post appliqué (Mode Threaded).")

def remove_style(style_id="custom-checkbox"):
    from IPython.display import Javascript, display
    
    return Javascript(f"""
    var s = document.getElementById("{style_id}");
    if (s) {{ s.remove(); }}
    """)

def ensure_style(css: str, style_id="custom-style"):
    from IPython.display import HTML, display
    display(HTML(f"""
    <script>
     var s = document.getElementById("{style_id}");
     if (s) {{ s.remove(); }}
     if (!document.getElementById("{style_id}")){{
        const style = document.createElement("style");
        style.id = "{style_id}";
        style.innerHTML = `{css}`;
        document.head.appendChild(style);
    }}
    </script>
    """))
    
def internetOk(URL): 
    print("Testing internet connexion...", end=' ')
    try:
        r = requests.get(URL, params={"key": "IOK33"})
        if r.status_code == 200:
            print("Connexion ok")
            return json.loads(r.content)['value']
        else:
            print(r.status_code) 
            if r.status_code == 404: 
                print("Bad URL")
            else:
                print("Connexion issue")
            return False
    except:
        print("No internet connexion")
        return False
    
    
class QuizLab:
    """
    LabQuiz2
    - Charger une banque de quizzes depuis YAML (from_yaml)
    - show(quiz_id) affiche toutes les questions du quiz (avec LaTeX)
    - Valider => score global
    - Tips => Affiche des indications pour orienter vers les bonnes réponses
    - Corriger => affiche tips correct/wrong par question
    - Reset => décoche toutes les cases et vide la sortie
    V2: changement du format d'import des questions
    """
    
    import ipywidgets as widgets
    from IPython.display import display, Javascript, Markdown
    from .utils import decode_dict_base64, calculate_quiz_score
    from .utils import get_full_object_hash
    
    _instance = None
    _registry = {}
    #_CHECKALIVE = 60

    
    def __init__(self,  URL="", QUIZFILE="", needAuthentification=True, retries=2,
                 exam_mode=False, test_mode=False, 
                 mandatoryInternet=False, CHECKALIVE=600,
                 in_streamlit=False):
        from .utils import StudentForm,  check_installed_package_integrity
        
        QUIZFILE_ORI = QUIZFILE
        self.in_streamlit = in_streamlit
        
        tic = time.perf_counter()
        #self.stop_event = threading.Event()
        if not self.in_streamlit:
            self.stop_event = asyncio.Event() # Remplace threading.Event
            self.stop_check_event = asyncio.Event() 
        self.quiz_bank = {}
        self.exam_mode = exam_mode
        self.test_mode = test_mode
        self.quiz_results = {}
        self.quiz_counts = {}
        self.quiz_correct = {}
        self.score_global = 0.0
        self.user_answers = {}
        self.keep_alive = True
        self.student = StudentForm()
        self.student.name = ""
        self.needAuthentification = needAuthentification
        self.encoded = False
        self.encrypted = False
        self.retries = retries
        self.sheetTransfer = True
        self.SHEET_URL = URL
        self.QUIZFILE = QUIZFILE if isinstance(QUIZFILE, str) else QUIZFILE.name
        self._CHECKALIVE = CHECKALIVE
        self.current_quiz_id = ""
        #self.question = ""
        self.machine_id = ""
        self.internetOK = ''
        
        self.style = widgets.HTML( 
            "<style>"
            ".correct { color:#008000; font-weight:bold; margin:4px 0; }"
            ".wrong { color:#b30000; font-weight:bold; margin:4px 0; }"
            ".tip { margin-left:20px; color:#444; font-style:italic }"
            "</style>"
        )
        
        #self.thread = threading.Thread(target=self.check_alive, daemon=True)
        patch_requests_post()
        
        self.internetOK = internetOk(URL) if mandatoryInternet else ''
        if mandatoryInternet and not self.internetOK:
            raise Exception("No internet connexion or bad URL")
        
        
        self.init()
        #print("ensure !")
        #print(self.checkbox_style)     
        ensure_style(self.checkbox_style, style_id="custom-checkbox")
        check_installed_package_integrity()
        
        if not QUIZFILE_ORI=="": 
            tic = time.perf_counter()
            self.load_from_yaml(QUIZFILE_ORI)
            toc = time.perf_counter()
            #print(f"Temps d'exécution load : {toc-tic:.3f} seconde(s)")
        
        # Ceci permet de ne créer qu'une seule instance logique 
        # (un seul thread de surveillance en réalité) pour un QUIZFILE donné
        # il peut y avoir plusieurs instances pour le même QUIZFILE, mais un seul thread
        # si autre QUIZFILE, autre thread de surveillance. 
        if QUIZFILE in QuizLab._registry:
            old = QuizLab._registry[QUIZFILE]
            old.stop() # Arrêt du thread
            # restaure les quiz déjà effectués
            self.retries = old.retries
            self.quiz_results = old.quiz_results
            self.quiz_counts = old.quiz_counts
            self.quiz_correct = old.quiz_correct
        
            
        QuizLab._registry[QUIZFILE] = self  
        
        if not self.in_streamlit:
            self.stop_event = asyncio.Event() # Remplace threading.Event
        self._task = None
        self._check_task = None
        self.start()
        toc = time.perf_counter()
        #print(f"Temps d'exécution __init__ : {toc-tic:.3f} seconde(s)")
    
    def hash(self):
        return get_full_object_hash(self, modules = ['main', 'utils'],
                             WATCHLIST=['exam_mode', 'test_mode', 'retries'])

        
    def testInternet(self):
        self.internetOK = internetOk(self.SHEET_URL) #
        if mandatoryInternet and not self.internetOK:
            raise Exception("No internet connexion or bad URL")
    
    def start(self):
        """Lance la surveillance sans bloquer"""
        # Dans JupyterLite/IPython, la boucle tourne déjà.
        # On crée une tâche non-bloquante.
        if not self.in_streamlit:
            self._task = asyncio.create_task(self.check_alive())


    def stop(self):
        if not self.in_streamlit:
            self.stop_event.set()
            if self._task:
                # Optionnel : pour forcer l'arrêt immédiat si nécessaire
                # self._task.cancel() 
                print("--> Redémarrage du quiz sous un nouveau nom")
                #print()

####################### NEW SHOW ############################
    checkbox_style = """
     /* base */
        .custom.widget-checkbox_bis input[type="checkbox"] {
            appearance: none;
            -webkit-appearance: none;
            -moz-appearance: none;
            width: 12px;
            height: 12px;
            top: 2px;
            border: 2px solid gray;
            border-radius: 4px;
            position: relative;
            cursor: pointer;
            color: black;
        }

        /* coche ✓ */
        .custom.widget-checkbox_bis input[type="checkbox"]:checked::after {
            content: "✓";
            position: absolute;
            top: -9px;
            left: -0px;
            font-size: 16px;
            font-weight: bold;
            color: currentColor; 
        } 
    
    
.custom.widget-checkbox input[type="checkbox"] {
    appearance: none;
    -webkit-appearance: none;
    -moz-appearance: none;

    width: 14px;
    height: 15px;
    border: 2px solid gray;
    border-radius: 4px;
    position: relative;
    cursor: pointer;

    box-sizing: border-box;
    vertical-align: middle;
}

/* coche */
.custom.widget-checkbox input[type="checkbox"]:checked::after {
    content: "";
    position: absolute;

    /* centrage géométrique */
    left: 50%;
    top: 40%; /* was 50% */
    transform: translate(-50%, -50%) rotate(35deg);

    /* proportions relatives */
    /*width: 5px;
    height: 9px;*/
    width: 35%;      /* largeur = 35% de la case */
    height: 70%;     /* hauteur = 50% de la case */


    border: solid currentColor;
    border-width: 0 2px 2px 0;
}

    
    
    /* new*/
    .custom.widget-checkbox_new input[type="checkbox"] {
    appearance: none;
    width: 12px;
    height: 12px;
    border: 2px solid gray;
    border-radius: 4px;
    position: relative;
    cursor: pointer;
    box-sizing: border-box;
}

.custom.widget-checkbox_new input[type="checkbox"]:checked::after {
    content: "";
    position: absolute;
    left: 3px;
    top: 0px;
    width: 3px;
    height: 6px;
    border: solid currentColor;
    border-width: 0 2px 2px 0;
    transform: rotate(45deg);
}
/*end new*/

        /* OK */
        .match.custom.widget-checkbox input[type="checkbox"] {
            border-color: green;
        }

        .match.custom.widget-checkbox input[type="checkbox"]:checked::after {
            color: green;
        }

        /* KO */
        .mismatch.custom.widget-checkbox input[type="checkbox"] {
            border-color: red;
         }
         
        .mismatch.custom.widget-checkbox input[type="checkbox"]:checked::after {
            color: red;
        }

        """


    def inject_css(self):
        # css adapté pour les checkboxes
        from IPython.display import display, HTML

        display(HTML(f"""
        <style>
        {self.checkbox_style}
        </style>
        """))
        
    def _getParameters(self):
        from .utils import get_source_integrity_hash, get_full_object_hash
        return  {
                    "exam_mode": self.exam_mode,
                    "test_mode": self.test_mode,
                    "retries": self.retries,
                    "counts": ",".join(str(v) for k, v in sorted(self.quiz_counts.items())),
                    "corrections": ",".join(str(v) for k, v in sorted(self.quiz_correct.items())),
                     "full_hash": get_full_object_hash(self, modules = ['main', 'utils'],
                             WATCHLIST=['exam_mode', 'test_mode', 'retries'])
                   }

        

    # -------------------------
    # Widgets de question
    # -------------------------
    def _make_question_widget(self, question_html):
        try:
            inner = widgets.HTMLMath(
                f"<div style='max-width:1800px; white-space: normal;'>{question_html}</div>"
            )
        except Exception:
            print("une exception dans le make_question_widget")            
            inner = widgets.HTML(
                f"<div style='max-width:1800px; white-space: normal;'>{question_html}</div>"
            )

        wrapper = widgets.HBox([inner])
        wrapper.layout = widgets.Layout(padding="0px", margin="0px", align_items="flex-start")
        inner.layout = widgets.Layout(padding="0px", margin="0px")
        return wrapper

    # -------------------------
    # Fabrique de widgets
    # -------------------------
    def build_answer_widgets(self, question, quiz_type, propositions):
        rows, widgets_list = [], []

        if quiz_type == "qcm":
            for prop in propositions:
                cb = widgets.Checkbox(value=False)
                cb.add_class("custom") 
                cb.layout = widgets.Layout(margin="0px", padding="0px")

                lbl = self._make_question_widget(prop["proposition"])
                lbl.layout = widgets.Layout(width="70%", margin="5px 0 0 -15%")

                row = widgets.HBox([cb, lbl])
                row.layout = widgets.Layout(align_items='flex-start', margin='4px 0', padding="0px")

                rows.append(row)
                widgets_list.append(cb)

        elif quiz_type == "numeric":
            def tolerance_text(prop):
                parts = []
                tol_pct = prop.get("tolerance", 0)      # tolérance relative
                tol_abs = prop.get("tolerance_abs", 0)  # tolérance absolue
                if tol_pct:
                    parts.append(f" ±{tol_pct*100}%")
                if tol_abs:
                    parts.append(f" ±{tol_abs}")

                tol_texte = "Tolérance : max de"+f"{' ou '.join(parts)}" if parts else ""
                return tol_texte

            for prop in propositions:
                tol_texte = tolerance_text(prop)
                if prop.get("type") == "int":
                    w = widgets.IntText(description=prop["proposition"])
                else:
                    w = widgets.FloatText(description=prop["proposition"])

                rows.append(widgets.HBox((w, widgets.HTML(tol_texte, 
                                                    layout=widgets.Layout(margin='0 0 0 20px') ))))
                widgets_list.append(w)

        return rows, widgets_list, question
    
    def compute_score(self, propositions, user_answers, quiz_type, constraints=None, weights=None): 
        from .utils import calculate_quiz_score

        score, total_possible = calculate_quiz_score(quiz_type, user_answers, 
                                propositions, constraints=constraints, weights=None)
        return score, total_possible

    def show(self, quiz_id, noscore=False, **context):
        from .utils import sanitize_dict
        """Affiche un quiz (QCM ou numérique) avec widgets indépendants"""
        from .utils import decode_dict_base64
        TYPE_MAP = {
        "int": int,
        "bool": lambda s: s.lower() == "true" if isinstance(s, str) else s,
        "float": float,
                    }
        # -------------------------
        # Sécurité & authentification
        # -------------------------
        
        if quiz_id not in self.quiz_bank:
            raise KeyError(f"Quiz '{quiz_id}' not found in bank.")
        
        if self.needAuthentification:
            if self.student is None or not self.student.name:
                print("⚠️ Authentification non effectuée -- Entrez vos nom et prénom !\nPuis ré-exécuter la cellule")
                self.authentification()
                return

        if self.quiz_counts[quiz_id] >= self.retries + 1:
            print("😔 Nombre maximum d'essais déjà atteint!!")
            return

        if self.quiz_correct[quiz_id] == 1:
            print("😔 Correction déjà obtenue pour ce quiz!!")
            return
        
        # -------------------------
        # Chargement du quiz 
        # -------------------------
        def _get_protected_data():
            """Récupère et déchiffre les données avec gestion d'erreurs."""
            import copy, os
            try:
                entry = self.quiz_bank[quiz_id]
                
                if self.encrypted:  #non encoded or encoded
                    from .utils import prep
                    from cryptography.fernet import Fernet, InvalidToken
                    try:
                        h = prep(4, quiz_id, os.path.splitext(self.QUIZFILE)[0], self.internetOK)
                        #frt = Fernet(h)
                        # Tentative de déchiffrement et chargement JSON
                        #decrypted_data = frt.decrypt(entry).decode('utf-8')
                        decrypted_data = Qwsp(h).weight(entry).decode('utf-8')
                        entry = json.loads(decrypted_data)
                    except (InvalidToken, ValueError) as e:
                        print(f"❌ Erreur de déchiffrement pour le quiz '{quiz_id}'.")
                        print("Vérifiez la clé d'accès ou l'intégrité des données.")
                        return None, None, None, None
                

                props = entry.get("propositions", []) or []
                if self.encoded:
                    props = decode_dict_base64(props)
                
                return copy.deepcopy(
                    (
                    entry.get("question", quiz_id), 
                    entry.get("type", "qcm"), 
                    props, 
                    entry.get("constraints", {})
                    )
                )
            
            except Exception as e:
                print(f"⚠️ Une erreur inattendue est survenue lors du chargement : {e}")
                return None, None, None, None
            
        
        # plus nécessaire
        def get_quiz(entry, encoded=False, encrypted=False):
            """."""

            if not encrypted: #non encoded or encoded
                question = entry.get("question", quiz_id)
                quiz_type = entry.get("type", "qcm")
                propositions = entry.get("propositions", {}) or {}
                constraints = entry.get("constraints", {}) or {}

            if encoded:
                propositions = decode_dict_base64(propositions)

            if encrypted:
                # ---------------
                from .utils import prep
                h = prep(4, quiz_id)
                from cryptography.fernet import Fernet, InvalidToken
                frt = Fernet(h)
                entry = json.loads(frt.decrypt(entry).decode('utf-8')) 
                question = entry.get("question", quiz_id)
                quiz_type = entry.get("type", "qcm")
                propositions = entry.get("propositions", {}) or {}
                constraints = entry.get("constraints", {}) or {}
                #dec = frt.decrypt(raw_bytes).decode('utf-8')
                # ---------------

            return question, quiz_type, propositions, constraints 


        entry = self.quiz_bank[quiz_id]
        question, quiz_type, propositions, constraints = None, None, None, None
        question, quiz_type, propositions, constraints = _get_protected_data()
        #print(question, quiz_type, propositions, constraints)

        allContainExpected = all( 'expected' in p for p in propositions )
        
        if quiz_type in ['numeric-template', 'qcm-template']:
            import numpy as np
            question = question.format(**context)
            for p in propositions:
                pexpect =  p.get("expected", '' if quiz_type=='qcm-template' else 0)
                ptype = p.get("type", bool if quiz_type == "qcm" else float)
                ptype = TYPE_MAP.get(ptype, ptype)
                preponse =  p.get("reponse",'')
                p["expected"] = ptype(eval(pexpect,{}, context)) if isinstance(pexpect, str) else pexpect
                if isinstance(preponse, str) and (preponse.startswith('f"') or preponse.startswith("f'")): 
                    p["reponse"] = str(eval(preponse,{},context))
                p['proposition'] = p['proposition'].format(**context)
            quiz_type = quiz_type.split('-')[0]
            #print(quiz_type, pexpect, propositions, "preponse", preponse, p["reponse"])
            
        #get_quiz(entry, encoded=self.encoded, encrypted=self.encrypted)

        random.shuffle(propositions)
        # Réponses par défaut
        self.user_answers[quiz_id] = {p["label"]:False if quiz_type=="qcm" else 0 for p in propositions} 


        rows, answer_widgets, question = self.build_answer_widgets(question, quiz_type, propositions)

        # -------------------------
        # Boutons
        # -------------------------
        btn_validate = widgets.Button(description="Valider", button_style="primary", icon="check")
        btn_reset    = widgets.Button(description="Reset", button_style="warning", icon="refresh")
        btn_tips     = widgets.Button(description="Tips", button_style="info", icon="check-circle")
        btn_correct  = widgets.Button(description="Corriger", button_style="success", icon="check-circle")

        btn_correct.layout.display = "none" if (self.exam_mode or self.test_mode) else "inline-flex"
        btn_tips.layout.display = "none" if self.exam_mode else "inline-flex"

        buttons = widgets.HBox([btn_validate, btn_reset, btn_tips, btn_correct])
        buttons.layout.margin = "8px 0 0 0"


        # -------------------------
        # Callbacks
        # -------------------------
        def on_validate(_):
            self.quiz_counts[quiz_id] += 1
            msg = ""
            user_answers = {p["label"]:w.value for p,w in zip(propositions,answer_widgets)} 
                                                            #on construit le dictionnaire 
                                                            #des labels:réponse utilisateur
            if context: user_answers['context'] = context
            self.user_answers[quiz_id] = user_answers
            
            if (not self.exam_mode) and allContainExpected:
                score, total = self.compute_score(propositions, user_answers, quiz_type, constraints=constraints, weights=None) #  compute_score()
                self.quiz_results[quiz_id] = score / total
                #msg += f"propositions {propositions}" 
                self.score_global = 20 * sum(self.quiz_results.values()) / len(self.quiz_results)

            msg += f"Essai n°{self.quiz_counts[quiz_id]} sur {self.retries+1}"

            if self.quiz_counts[quiz_id] >= self.retries + 1:
                msg += "<br><b>😔 Nombre maximum d'essais atteint</b>"
                for btn in [btn_validate, btn_tips, btn_reset]:
                    btn.disabled = True
                    btn.icon = "ban"
            
            if (not self.exam_mode) and (not noscore) and allContainExpected:
                msg += f"<h3>Score : <b>{score}/{total}</b></h3>"
                msg += "🎉 Bravo !" if score == total else "⚠️ Tout n'est pas correct."
            if (not self.exam_mode) and (not allContainExpected):
                msg += "<br>⚠️ pas de réponses dans le fichier de quiz: pas de calcul du score."

            output.clear_output()
            with output:
                display(widgets.HTML(msg))

            if self.sheetTransfer:
                """if quiz_type == "qcm":
                    answers = {p.get("label"): w.value for w, p in zip(answer_widgets, propositions)}
                else:
                    answers = {propositions[0].get("label"): answer_widgets[0].value}"""
                #answers = {p.get("label"): w.value for w, p in zip(answer_widgets, propositions)}
                #if context: answers['context'] = context
                
                try:
                    score = 0 if self.exam_mode else score / total
                except NameError:
                    score = 0
                    
                self.record_event(
                    event_type="validate_exam" if self.exam_mode else "validate",
                    quiz_id=quiz_id,
                    parameters = self._getParameters(),
                    answers=user_answers, #answers, 
                    score=score
                )

        def on_reset(_):
            for w in answer_widgets:
                if hasattr(w, "value"):
                    w.value = False if quiz_type == "qcm" else 0
            output.clear_output()

        def on_tips(_):
            output.clear_output()
            with output:
                if quiz_type == "qcm":
                    for w, p in zip(answer_widgets, propositions):
                        if w.value != p.get("expected", False) and p.get("tip"):
                            display(Markdown(f"ℹ️ **{p.get('label','')}** — {p['tip']}"))
                else:
                    for w, p in zip(answer_widgets, propositions):
                        diff = abs(w.value - p.get("expected", 0))
                        tol = max(p.get("tolerance_abs", 0), 
                              p.get("tolerance", 0) * abs(p.get("expected", 0)))
                        
                        if (diff> tol) and p.get("tip"):
                            display(Markdown(f"ℹ️ **{p.get('label','')}** — {p['tip']}"))
                        """tip = p.get("tip")
                        if tip:
                             #display(Markdown(f"ℹ️ {tip}"))"""
                        
        def corriger(_=None):
            for cb in checkboxes:
                cb.disabled = True
                cb.remove_class("match")
                cb.remove_class("mismatch")

                if cb.value == cb.expected:
                    cb.add_class("match")
                else:
                    cb.add_class("mismatch")


        def on_correct(_):
            
            if not allContainExpected:
                msg = "<br>⚠️ pas de réponses dans le fichier de quiz: pas de correction possible."
                output.clear_output()
                with output:
                    display(widgets.HTML(msg))
                return
                        
            score, total = self.compute_score(propositions, self.user_answers[quiz_id], quiz_type, constraints=constraints, weights=None) #  compute_score()
            msg = f"<h3>Score : <b>{score}/{total}</b></h3>"
            output.clear_output()
            with output:
                display(widgets.HTML(msg))
                if quiz_type == "qcm":
                    for r, w, p in zip(rows, answer_widgets, propositions):
                        #w.value = p.get("expected", False)
                        #w = r.children[0]
                        w.remove_class("match")
                        w.remove_class("mismatch")
                        w.add_class("custom")
                        if w.value == p.get("expected", False):
                            w.add_class("match")
                        else:
                            w.add_class("mismatch")
                        
                        newlbl = self._make_question_widget(f"<b>{p.get('label','')}</b> - " + p["proposition"])
                        newlbl.layout = widgets.Layout(width="70%", margin="5px 0 0 -15%")
                        r.children = [r.children[0], newlbl]
                        if p.get("reponse"):
                            display(Markdown(f"{'✅' if p.get('expected') else '❌'} **{p.get('label','')}** — {p['reponse']}"))
                    if constraints:
                        for c in constraints:
                            if c["type"] == "XOR": 
                                display(Markdown(f"⚠️ Les réponses à **{c['indices']}** sont forcément différentes"))
                            elif c["type"] == "SAME": 
                                display(Markdown(f"⚠️ Les réponses à **{c['indices']}** sont forcément identiques"))
                            elif c["type"] == "IMPLY": 
                                display(Markdown(f"⚠️ La réponse **{c['indices'][0]}** vraie implique que **{c['indices'][1]}** est vraie"))
                            elif c["type"] == "IMPLYFALSE": 
                                display(Markdown(f"⚠️ La réponse **{c['indices'][0]}** vraie implique que **{c['indices'][1]}** est nécessairement fausse"))
                else:
                    for w, p in zip(answer_widgets, propositions):
                        """pexpect =  p["expected"]
                        w.value = eval(pexpect,{},context) if isinstance(pexpect, str) else pexpect"""
                        w.value = p["expected"] # ce qui précède pour extension future
                        rep = p.get("reponse")
                        #if rep:
                        #    display(Markdown(rep))

            self.quiz_correct[quiz_id] = 1
            for btn in [btn_validate, btn_tips, btn_reset]:
                btn.disabled = True
                btn.icon = "ban"
            
            if self.sheetTransfer:

                user_answers = {p["label"]:w.value for p,w in zip(propositions,answer_widgets)} 
                if context: user_answers['context'] = sanitize_dict(context)
                
                self.record_event(event_type="correction", quiz_id=quiz_id, parameters=self._getParameters(),
                                  answers=user_answers, score=1)

        # -------------------------
        # Binding
        # -------------------------
        btn_validate.on_click(on_validate)
        btn_reset.on_click(on_reset)
        btn_tips.on_click(on_tips)
        btn_correct.on_click(on_correct)

        # -------------------------
        # Affichage final
        # -------------------------
        output = widgets.Output()
        container = widgets.VBox([
            self.style,
            widgets.HTMLMath(f"<h3>{question}</h3>"),  #.format(**context)), Pour extension future 
            widgets.VBox(rows),
            widgets.HTML("<div style='height:20px'></div>"),
            buttons,
            output
        ])

        output.clear_output()
        display(container)

        display(Javascript("""
            if (window.MathJax && MathJax.typesetPromise) {
                MathJax.typesetPromise();
            }
        """))

        #return container
    

    def exam_show(self, exam_title="", questions=None, shuffle=False, nb=None):
        """
        Affiche un examen à partir d’un ensemble de questions et retourne
        la liste des questions sélectionnées.

        Parameters
        ----------
        exam_title: str, default ""
            Utilisé pour identifier l'examen
        questions : list[str] or None
            Liste de labels de questions (ex. "quiz1", "quiz2").
            Si None, utilise toutes les questions de la banque.
        shuffle : bool, default False
            Si True, mélange l’ordre des questions avant l’affichage.
        nb : int or None
            Si différent de None, tire aléatoirement nb questions distinctes
            parmi l’ensemble des questions disponibles.

        Returns
        -------
        list[str]
            Liste des labels des questions effectivement affichées,
            dans l’ordre de présentation.
        """
        from IPython.display import display, Markdown, HTML
        from ipywidgets import Output

        # 1. Base de questions
        if questions is None:
            qlist = list(self.available_quizzes())
        else:
            qlist = list(questions)

        # 2. Tirage aléatoire de nb questions
        qlist_out = qlist
        if nb is not None:
            if nb > len(qlist):
                raise ValueError("nb est supérieur au nombre de questions disponibles")
            qlist_out = []  # si tirage de nb questions dans qlist, on stockera ces questions dans qlist_out
                            # car certaines questions (avec contexte) ne sont pas affichables
                            # on ne peut pas trier de base sur les types, car on n'y a pas accès 
                            # à ce niveau dans le cas crypté
        else:
            nb = len(qlist)
            #qlist = random.sample(qlist, nb)"""

        # 3. Mélange si demandé
        if shuffle:
            random.shuffle(qlist)

        # 4. Affichage de l'examen
        if exam_title != "": 
            display(HTML(f"<h2> {exam_title} </h2>"))
        m = 0
        for n, label in enumerate(qlist):
            try:                
                display(HTML(f"<h3> Question {m+1} </h3>"), display_id="q_"+str(n)) # ({label})
                self.show(label,noscore=True)
                qlist_out.append(label)
                m = m + 1
                if len(qlist_out) == nb: break
            except: 
                # Efface le libellé de question précédent
                update_display(HTML(''), display_id="q_"+str(n))
        
        # 5. Retourner la liste des questions affichées
        if self.sheetTransfer:  
            exam = exam_title if exam_title != "" else "exam"
            self.record_event(event_type="starting", quiz_id=exam, parameters=self._getParameters(),
                              answers=qlist_out, score=0)
        return qlist_out
    
    def exam_result(self, qlist, bareme=None):
        allquizzes = [q for q in self.quiz_bank.keys() if q != "title"]
        if bareme is not None: 
            bareme = {q:1 if q not in bareme.keys() else bareme[q] for q in allquizzes }
        else:
            bareme = {q:1 for q in allquizzes}
        for n,quiz_id in enumerate(qlist):    
            print(f"Question {n} : {self.quiz_results[quiz_id]*bareme[quiz_id]} sur {bareme[quiz_id]}")
        print('-'*22)    
        ponder = [ bareme[q]*self.quiz_results[q]  for q in qlist] 
        bareme_exam = [ bareme[q]  for q in qlist] 
              
        print("Note sur 20 : ", sum(ponder)/sum(bareme_exam)*20)
                  

    # ---------------------------
    # YAML loader / helpers
    # ---------------------------

    
    @classmethod
    def from_yaml(cls, path, default_quiz_id=None):
        """
        Crée une instance et charge la banque depuis un fichier YAML.
        Si encoded=True, décode le fichier Base64 avant parsing.
        :param path: chemin vers quiz_bank.yaml
        :param default_quiz_id: (optionnel) id du quiz à sélectionner par défaut
        """
        
        from .utils import is_base64_encoded_text
        tic = time.perf_counter()
        inst = cls()
        with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        
        inst.quiz_bank = data
        # Détecte si encoded sur la première question
        cle = list(data.keys())[0]
        q = data[cle]["propositions"][0] 
        inst.encoded = is_base64_encoded_text(q)   
        
        # initialise les résultats pour l'ensemble des quiz_id
        inst.quiz_counts = {quiz_id:0 for quiz_id in inst.quiz_bank}
        inst.quiz_results = {quiz_id:0 for quiz_id in inst.quiz_bank}
        inst.quiz_correct = {quiz_id:0 for quiz_id in inst.quiz_bank}
        toc = time.perf_counter()
        #print(f"Temps d'exécution from_yaml : {toc-tic:.3f} seconde(s)")
        return inst

    def authentification(self):
        from .utils import compute_machine_id
        tic = time.perf_counter()
        self.student.display()  
        toc = time.perf_counter()
        #print(f"Temps d'exécution authentification : {toc-tic:.3f} seconde(s)")
  


    def record_event(self, event_type, quiz_id, parameters, answers, score):
        """Envoie une ligne dans Google Sheets."""
        
        payload = {
            "notebook_id": self.machine_id,
            "student": self.student.name,
            "quiz_title": quiz_id,
            "timestamp": datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"), #.isoformat(timespec="seconds"),
            "event_type": event_type,           # "validate" ou "correct"
            "parameters": parameters,
            "answers": answers,                 # dict {label: bool}
            "score": str(score)
        }
        
        try:
            #requests.post(self.SHEET_URL, json=payload)
            r = requests.post(
            #r = await post_data(
            self.SHEET_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "text/plain"}
            )
            #print(payload)
        except TypeError as e:
            print("⚠️ Attention aux paramètres passés qui ne sont pas json-sérialisable")
            print("Simplifier ou convertir", e)
        except Exception as e:
            print("⚠️ Erreur d’envoi :", e)
            
  

    async def check_alive(self):
        """Version asynchrone (pour éviter les threads --> pyodide"""
        from .utils import get_big_integrity_hash
        
        while not self.stop_event.is_set():
            if self.keep_alive:
                parameters = self._getParameters()
                big_hash = get_big_integrity_hash(self, modules = ['main', 'utils'],
                             WATCHLIST=['exam_mode', 'test_mode', 'retries'])
                parameters['get_big_integrity_hash'] = big_hash
                self.record_event("check_integrity", "integrity", parameters,"", 0)
            
            # Sleep asynchrone qui ne bloque pas le navigateur
            try:
                # Attend 600s OU que stop_event soit activé
                await asyncio.wait_for(self.stop_event.wait(), timeout=self._CHECKALIVE)
            except asyncio.TimeoutError:
                # Si le timeout expire, on continue simplement la boucle while
                pass

                
    def check_integrity(self):
        import inspect
        print("Source:\n", inspect.getsource(self.check_integrity))        
        check = {
            "exam_mode": self.exam_mode,
            "authentification": self.needAuthentification,
            "transfer": self.sheetTransfer,
            "retries": self.retries
        }
        payload = {
            "notebook_id": self.machine_id,
            "student": self.student.name,
            "quiz_title": "integrity",
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "event_type": "teacher_check",    
            "answers": check,                
            "score": 0
        }
        try:
            #requests.post(self.SHEET_URL, json=payload)
            r = requests.post(
            self.SHEET_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "text/plain"}
            )
        except Exception as e:
            print("⚠️ Erreur d’envoi :", e)


        
    def _make_the_check(self, pw, ARCHIVE_NAME, WORKDIR):
        from .utils import compute_local_hash, _fernet_key_from_password, _sha256_bytes
        import shutil, tarfile

        # --- clé Fernet ---
        key = _fernet_key_from_password(pw.value)
        fernet = Fernet(key)

        # --- lecture + déchiffrement ---
        encrypted = Path(ARCHIVE_NAME).read_bytes()
        try:
            tar_bytes = fernet.decrypt(encrypted)
        except:
            raise KeyError("Mot de passe incorrect")

        # --- extraction ---
        if WORKDIR.exists():
            shutil.rmtree(WORKDIR)
        WORKDIR.mkdir()
        with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
            tar.extractall(WORKDIR)

        sys.path.insert(0, str(WORKDIR))
        import importlib.util
        chemin_fichier = str(WORKDIR) + "/zutils.py"
        nom_module = "zutils"

        # Charger le module dynamiquement
        spec = importlib.util.spec_from_file_location(nom_module, chemin_fichier)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # On lance un daemon de surveillance, après avoir arrêté le premier
        #module.do_the_check(self, WORKDIR)
        self._check_task = asyncio.create_task(module.do_the_check(self, WORKDIR))
        #self.stop_check_event = asyncio.Event()    

        # --- nettoyage ---
        del module
        #shutil.rmtree(WORKDIR)
        print("✅ Vérification effectuée, données envoyées, daemon lancé")
        print(f"✅ Suppression de {WORKDIR}")

            
    def check_quiz(self, archive="quiz.tar.enc"):
        from .utils import compute_local_hash, _fernet_key_from_password, _sha256_bytes
        import tempfile

        ARCHIVE_NAME = archive
        tmpdir = tempfile.mkdtemp()
        WORKDIR = Path(tmpdir) 
        
        pw = widgets.Password(description="Mot de passe")
        btn = widgets.Button(description="Déverrouiller", button_style="primary")
        out = widgets.Output()

        display(widgets.VBox([pw, btn,out]))

        def on_submit(_):
            with out:
                out.clear_output()
                try:
                    if not pw.value:
                        raise ValueError("Mot de passe vide")
                    self._make_the_check(pw, ARCHIVE_NAME, WORKDIR)

                except Exception as e:
                    display(Markdown(f"❌ Erreur : `{e}`"))

        btn.on_click(on_submit)
    
    def load_from_yaml(self, path, default_quiz_id=None):
        """
        Charge la banque depuis un fichier YAML.
        Si encoded=True, décode le fichier Base64 avant parsing.
        :param path: chemin vers quiz_bank.yaml
        :param default_quiz_id: (optionnel) id du quiz à sélectionner par défaut
        """
        
        from .utils import is_base64_encoded_text, is_encrypted

        if isinstance(path, BytesIO):
            data = yaml.safe_load(path)
        else:
            with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
        self.quiz_bank = data
        # Détecte si encrypted sur le titre
        #print("data.keys()", data.keys())
        if "title" in data.keys():  
            self.encrypted = is_encrypted(data["title"])
        else:
            # Détecte si encoded sur la première question
            cle = list(data.keys())[0]
            q = data[cle]["propositions"][0] 
            self.encoded = is_base64_encoded_text(q)
        
        # initialise les résultats pour l'ensemble des quiz_id
        self.quiz_counts = {quiz_id:0 for quiz_id in self.quiz_bank}
        self.quiz_results = {quiz_id:0 for quiz_id in self.quiz_bank}
        self.quiz_correct = {quiz_id:0 for quiz_id in self.quiz_bank}
        self.user_answers = {quiz_id:{} for quiz_id in self.quiz_bank if quiz_id != "title"}


         
    def __load_quiz(self, quiz_id): 
        """Charge en mémoire le quiz identifié par quiz_id (ne l'affiche pas)."""
        if quiz_id not in self.quiz_bank:
            raise KeyError(f"Quiz '{quiz_id}' not in bank.")
        entry = self.quiz_bank[quiz_id]
        if not self.encrypted: #non encoded or encoded
            question = entry.get("question", quiz_id)
            quiz_type = entry.get("type", "qcm")
            propositions = entry.get("propositions", {}) or {}
            constraints = entry.get("constraints", {}) or {}

        if self.encoded:
            propositions = decode_dict_base64(propositions)
            
        if self.encrypted:
            raise TypeError("Le fichier ne doit pas être encrypté")
            
        return question, quiz_type, propositions, constraints
        
    
    
    def init(self):
        from .utils import compute_machine_id, get_source_integrity_hash, get_full_object_hash
        
        self.machine_id = compute_machine_id()
        if self.SHEET_URL=="":
            self.keep_alive = False
            self.sheetTransfer = False
        else:
            if self.keep_alive:
                #self.init = True
                #print("Waiting for network...", end='')
                parameters = self._getParameters()
                parameters["src_hash"] = get_source_integrity_hash(self.__class__)
                self.record_event("starting", "starting", parameters, "", 0)
                #print("Ready")
        if self.needAuthentification:
            self.authentification()


    # ---------------------------
    # convenience
    # ---------------------------

    def set_exam_mode(self, value=True):
        """Active/désactive le mode examen (cache le bouton Corriger et Tips si True)."""
        self.exam_mode = bool(value)
        if value: self.test_mode = not bool(value)
        
    def set_test_mode(self, value=True):
        """Active/désactive le mode Test (cache le bouton Corriger si True)."""
        self.test_mode = bool(value)
        if value: self.exam_mode = not bool(value) 
        
    def set_sheet_url(self, url=""):
        self.SHEET_URL = url

    def available_quizzes(self):
        """Retourne la liste des quiz ids disponibles dans la bank."""
        return [key for key in self.quiz_bank.keys() if key != 'title']


    def get_propositions(self, quiz_id):
        """Retourne la liste des labels (propositions) du quiz courant."""
        propositions = self._load_quiz(quiz_id)
        return list(self.propositions.keys())

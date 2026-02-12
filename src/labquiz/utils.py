# %load LabQuiz/utils.py
import yaml, json, base64, sys
import ipywidgets as widgets
from IPython.display import display, Markdown
import hashlib, inspect, re, types
from importlib.metadata import metadata
from pathlib import Path
import numpy as np
import pandas as pd
import inspect

from types import ModuleType
from typing import Iterable, Tuple




def b64_encode(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")

def b64_decode(text: str) -> str:
    return base64.b64decode(text.encode("ascii")).decode("utf-8")


def decode_dict_base64_(s: str) -> dict:
    json_str = base64.b64decode(s[0].encode("ascii")).decode("utf-8")
    return json.loads(json_str)

def decode_dict_base64(encoded_list):
    decoded = []
    for elt in encoded_list:
        json_str = base64.b64decode(elt.encode("ascii")).decode("utf-8")
        decoded.append(json.loads(json_str))
    return decoded

def is_base64_encoded_text(s: str) -> bool:
    try:
        decoded = base64.b64decode(s, validate=True).decode("utf-8")
    except Exception:
        return False

    # Heuristique : texte imprimable
    printable_ratio = sum(c.isprintable() for c in decoded) / len(decoded)
    return printable_ratio > 0.95


def get_macaddress():
    import uuid
    mac = uuid.getnode()
    mac_str = ':'.join(f'{(mac >> ele) & 0xff:02x}' for ele in range(40, -1, -8))
    if (mac >> 40) % 2: mac_str = "00:00:00:00:00:00" #“Probably generated (randomized) MAC address”
    return mac_str

def getUser():
    import os, uuid
    WE_ARE_IN_JUPYTERLITE = "pyodide" in sys.modules or "piplite" in sys.modules
    if WE_ARE_IN_JUPYTERLITE:
        try:
            id_file = ".labquiz_user_id"
            if os.path.exists(id_file):
                # The existing ID is read
                with open(id_file, "r") as f:
                    user_id = f.read().strip()
            else:
                # A new unique ID is generated
                user_id = str(uuid.uuid4())
                with open(id_file, "w") as f:
                    f.write(user_id)
                # Note: In JupyterLite, the file system is 
                # # automatically synced with browser storage.                
                # # Note : Dans JupyterLite, le système de fichiers est 
                # automatiquement synchronisé avec le stockage du navigateur.
        except Exception as e:
            user_id = "erreurUser"            
        return user_id
    else:   # We are not in JupyterLite
        import os
        try:
            user = os.getlogin()
        except:
            user = "erreurUser"
    return user
    
def compute_machine_id():
    import platform
    import hashlib
    raw = "|".join([
        platform.node(),
        platform.system(),
        platform.release(),
        platform.machine(),
        platform.processor(),
        #platform.python_version(),
        getUser(),
        #get_macaddress(),
    ])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


import ipywidgets as widgets
from IPython.display import display, Markdown

import ipywidgets as widgets
from IPython.display import display, Markdown


class StudentForm:
    def __init__(self):
        self.name = None
        self.output = widgets.Output()

        self.student_lastname = widgets.Text(
        placeholder="FAMILY NAME",
        description="FAMILY NAME:",
        style={'description_width': '70px'},
        layout=widgets.Layout(width="250px"),
        continuous_update=True
    )   
        self.student_firstname = widgets.Text(
        placeholder="First name",
        description="First name:",
        style={'description_width': '70px'},
        layout=widgets.Layout(width="250px"),
        continuous_update=True
    )
        self.save_name_button = widgets.Button(
        description="Save",
        button_style="info",
        icon="check"
    )
    
        self.student_firstname.observe(self.validate, "value")
        self.student_lastname.observe(self.validate, "value")
        
        self.save_name_button.on_click(self.on_save)
        self.student_firstname.on_submit(self.on_submit) 
        self.student_lastname.on_submit(self.on_submit)
        
        self.validate(None)  # Attend la synchro - disabled au départ


    def on_submit(self, _):
        self.on_save(None)
    
    
    def validate(self, _):
        self.save_name_button.disabled = not (
            self.student_firstname.value.strip()
            and self.student_lastname.value.strip()
        )

    def on_save(self, _):
        self.name = (
            self.student_lastname.value.strip()
            + " "
            + self.student_firstname.value.strip()
        )
        if len(self.name.strip()) <= 1:
            import time
            time.sleep(0.3)
            self.onsave()
        
        with self.output:
            self.output.clear_output()
            display(Markdown(f"✔️ **Name recorded** `{self.name}`"))

    
    def display(self):
        form = widgets.VBox([
            widgets.HTML("<h2 style='margin-top: 0; margin-bottom: 0; line-height: 1.2;'>Enter your first name and LAST NAME here </h2>"),
            #widgets.HTML("<b> &nbsp;&nbsp;&nbsp; ⚠️ (appuyez sur Entrée pour valider) </b>"),
            widgets.HBox([self.student_firstname, self.student_lastname]),
            self.save_name_button,
            self.output
           ])
        display(form)

###


def sanitize_dict(d):
    out = {}
    if any( inspect.ismodule(v) for v in d.values() ):
        out = {"modules":{}}

    for k, v in d.items():

        # Modules
        if isinstance(v, types.ModuleType):
            out["modules"][k] = v.__name__

        # NumPy arrays
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()

        # NumPy scalars
        elif isinstance(v, np.generic):
            out[k] = v.item()

        # Pandas
        elif isinstance(v, (pd.DataFrame, pd.Series)):
            out[k] = v.to_dict() if hasattr(v, "to_dict") else v.tolist()

        # Other non serializable objects → string. // Autres objets non sérialisables → string
        elif hasattr(v, "__dict__"):
            out[k] = str(v)

        #  JSON native types
        else:
            out[k] = v

    return out



###
def get_source_integrity_hash(cls):
    """
    Computes a hash based on the source text of the class's methods.
    """
    try:
        # 1. Entire class source code
        source = inspect.getsource(cls)
        
        # 2. Cleanup for more stability: removing comments, empty lines

        source = re.sub(r'#.*', '', source) 
        source = "\n".join([line.strip() for line in source.splitlines() if line.strip()])
        
        # 3. Hashing
        return hashlib.sha256(source.encode('utf-8')).hexdigest()
    
    except (OSError, TypeError):
            return "Source non disponible"



def get_ultra_integrity_hash(cls):
    """
    Calculates a hash based on bytecode AND constants (fixed values) 
    methods defined locally in the class.
    """
    method_hashes = []

    for name, attr in cls.__dict__.items():
        func = None
        if inspect.isfunction(attr):
            func = attr
        elif isinstance(attr, (staticmethod, classmethod)):
            func = attr.__func__

        if func and hasattr(func, "__code__"):
            code_obj = func.__code__
            
            # 1. bytecode
            bytecode = code_obj.co_code
            
            # 2. Constants (hardcoded values: strings, int, etc.) 
            # # We convert to string to be able to hash it easily
            consts = str(code_obj.co_consts).encode('utf-8')
            
            # We combine the name, bytecode and constants
            signature = f"{name}:".encode('utf-8') + bytecode + consts
            method_hashes.append(hashlib.sha256(signature).hexdigest())

    if not method_hashes:
        return None

    method_hashes.sort()
    final_payload = "".join(method_hashes).encode('utf-8')
    return hashlib.sha256(final_payload).hexdigest()



####### Yet another new version of Hash ############

# Hash independent of python version
# broken down into small modules
# - initial source code
# - living object
# - watchl#ist
# dependencies used
# With module source browsing
# bytecodes deleted

# ---------- Basic functions ----------

def hash_fonction(func: callable) -> bytes:
    """
    Hash robuste d'une fonction ou méthode.
    """
    h = hashlib.sha256()

    # Source if available
    try:
        src = inspect.getsource(func)
        h.update(src.encode("utf-8"))
    except (OSError, IOError):
        pass

    """code = func.__code__
    h.update(code.co_code)
    h.update(repr(code.co_consts).encode("utf-8"))
    h.update(repr(code.co_names).encode("utf-8"))"""

    return h.digest()

def hash_fonction_new(func: callable) -> bytes: #WAS #hash_callable_live(func) -> bytes:
    h = hashlib.sha256()

    # Linked method case → retrieve the real function
    func = getattr(func, "__func__", func)

    # Pure Python function
    if hasattr(func, "__code__"):
        code = func.__code__

        h.update(code.co_code)
        h.update(repr(code.co_consts).encode())
        h.update(repr(code.co_names).encode())
        h.update(repr(code.co_varnames).encode())
        h.update(repr(code.co_freevars).encode())
        h.update(repr(code.co_cellvars).encode())
        h.update(str(code.co_argcount).encode())
        h.update(str(code.co_kwonlyargcount).encode())
        h.update(str(code.co_flags).encode())

        # localization runtime (IPython use it)
        h.update(code.co_filename.encode())
        h.update(str(code.co_firstlineno).encode())

    else:
        # builtin / C-extension
        h.update(repr(func).encode())

    return h.digest()


# ---------- First level functions / Fonctions de premier niveau ----------

def fonctions_premier_niveau(module: ModuleType) -> Iterable[Tuple[str, callable]]:
    for name, obj in vars(module).items():
        if inspect.isfunction(obj) and (obj.__module__ == module.__name__ or obj.__module__ == '__main__'):
            yield name, obj


# ---------- Classes ----------

def methodes_de_classe_old(cls) -> Iterable[Tuple[str, callable]]:
    """
    Returns methods defined directly within the class (not inherited).
    """
    for name, obj in cls.__dict__.items():
        # staticmethod / classmethod
        if isinstance(obj, (staticmethod, classmethod)):
            yield name, obj.__func__
        elif inspect.isfunction(obj):
            #print("name", name)
            yield name, obj

def methodes_de_classe(cls):
    """
    Returns methods defined directly within the class (not inherited, not imported).
    """
    for name, obj in cls.__dict__.items():

        # unwrap staticmethod / classmethod
        if isinstance(obj, (staticmethod, classmethod)):
            func = obj.__func__
        elif inspect.isfunction(obj):
            func = obj
        else:
            continue

        if not func.__qualname__.startswith(cls.__name__ + "."):
            continue

        yield name, func


def methodes_de_classe_plus(cls):
    """
    Return methods defined directly in the class (not inherited, not imported).
    """
    for name, obj in cls.__dict__.items():

        # 1. Extraire l'objet fonction réel
        if isinstance(obj, (staticmethod, classmethod)):
            func = obj.__func__
        elif inspect.isfunction(obj):
            func = obj
        else:
            continue

        #2. Eliminate functions coming from another module
        if func.__module__ != cls.__module__:
            continue

        #3. Eliminate non-lexically defined functions in the class
        if not func.__qualname__.startswith(cls.__name__ + "."):
            continue
        
        yield name, func


def hash_classe(cls) -> bytes:
    """
    Hash of a class based on its methods and its name.
    """
    h = hashlib.sha256()

    # Name + bases
    h.update(cls.__name__.encode("utf-8"))
    for base in cls.__bases__:
        h.update(base.__name__.encode("utf-8"))

    # Methods
    for name, method in sorted(methodes_de_classe(cls), key=lambda x: x[0]):
        h.update(name.encode("utf-8"))
        h.update(hash_fonction(method))
        #print(name, "-->", h.hexdigest())

    return h.digest()


def classes_du_module(module: ModuleType) -> Iterable[Tuple[str, type]]:
    for name, obj in vars(module).items():
        if inspect.isclass(obj) and obj.__module__ == module.__name__:
            yield name, obj


# ---------- Hash global du module ----------

def hash_module(module: ModuleType) -> str:
    h = hashlib.sha256()

    # Fonctions
    for name, func in sorted(fonctions_premier_niveau(module), key=lambda x: x[0]):
        h.update(b"F")
        h.update(name.encode("utf-8"))
        h.update(hash_fonction(func))
        #print(name, "-->", h.hexdigest())

    # Classes
    for name, cls in sorted(classes_du_module(module), key=lambda x: x[0]):
        h.update(b"C")
        h.update(name.encode("utf-8"))
        h.update(hash_classe(cls))

    return h.hexdigest()

def hash_methodes_vivantes(obj) -> str:
    h = hashlib.sha256()

    for name, func in sorted(methodes_vivantes_objet(obj)):
        h.update(b"M")
        h.update(name.encode())
        h.update(hash_fonction(func))
        
    return h.hexdigest()

## This works!

def methodes_vivantes_objet(obj):
    cls = obj.__class__
    module_name = cls.__module__

    for name in dir(obj):
        if name.startswith("__"):
            continue

        try:
            attr = getattr(obj, name)
        except Exception:
            continue

        if not callable(attr):
            continue

        # bound method → actual function
        func = getattr(attr, "__func__", attr)

        # 1. Exclude external imports
        if func.__module__ != module_name:
            continue

        #2. Exclude external inherited methods
        if not func.__qualname__.startswith(cls.__name__ + "."):
            continue

        yield name, func

def hash_classe_vivante(cls) -> bytes:
    h = hashlib.sha256()

    for name in dir(cls):
        if name.startswith("__"):
            continue

        try:
            attr = getattr(cls, name)
        except Exception:
            continue

        if not callable(attr):
            continue

        func = getattr(attr, "__func__", attr)

        if func.__module__ != cls.__module__:
            continue

        if not func.__qualname__.startswith(cls.__name__ + "."):
            continue

        h.update(name.encode())
        h.update(hash_fonction(func))

    return h.digest()

def hash_dependances_modules(obj, modules_cibles) -> bytes:
    """
    Hash of global functions in target_modules used by class methods.

    obj: class whose graph we want to secure
    target_modules: list of module names to include (ex: ["main", "utils"])
    """
    
    cls = obj.__class__
    module_name = cls.__module__
    package = obj.__module__.split('.', 1)[0]
    
    h = hashlib.sha256()
    # extract dependencies used by class methods
    dependances = set()
    for name, func in methodes_vivantes_objet(obj):
        dependances |= set(func.__code__.co_names)
    #print(dependances)
    # browse targeted modules only
    for module in modules_cibles:

        for name in sorted(dependances):
            if not hasattr(module, name):
                continue
            obj = getattr(module, name)
            #print("pre-test", h.hexdigest())
            # ne traiter que les fonctions Python
            if isinstance(obj, (types.FunctionType, types.MethodType)):
                if obj.__module__ != module.__name__:
                    continue
                #h.update(name.encode())
                #h.update(hash_methodes_vivantes(obj).encode())
                h.update(hash_fonction(obj))
                #print("dependances:", name, hash_fonction(obj).hex() )
                #print("  -->  ", h.hexdigest())
            else:
                pass
                # if necessary, we can also hash other objects, for example classes or constants
                #h.update(name.encode())
                #h.update(repr(obj).encode())
     
    return h.hexdigest()


# --------- watchlist --------------------

def stable_value(val):
    if val is None or isinstance(val, (int, float, bool, str)):
        return repr(val)
    if isinstance(val, (list, tuple)):
        return "[" + ",".join(stable_value(v) for v in val) + "]"
    if isinstance(val, dict):
        return "{" + ",".join(
            f"{stable_value(k)}:{stable_value(v)}"
            for k, v in sorted(val.items(), key=lambda x: repr(x[0]))
        ) + "}"
    return f"<UNHASHABLE:{type(val).__name__}>"


def hash_watchlist(obj, WATCHLIST=[]):
    cfg_parts = []
    for name in WATCHLIST:
        if hasattr(obj, name):
            val_str = stable_value(getattr(obj, name))
            cfg_parts.append(f"{name}:{val_str}")
    cfg_parts.sort()
    payload = "|".join(cfg_parts)
    return hashlib.sha256(payload.encode()).hexdigest()

def get_big_integrity_hash(obj, modules=[], WATCHLIST=[]):
    """
    Returns a dict with: 
    - "source_hash": modules_hash, # Sources 
    - "watchlist_hash": watchlist_hash, # monitored parameters 
    - "live_object":living_object, # living object methods -- identifies monkey patching on the object 
    - "dependencies": dependencies, # the living methods of the modules used by the current object 
    - "full_hash": full_hash
    """
    
    modules_hash = []
    package = obj.__module__.split('.', 1)[0]
    modules = [sys.modules[package+"."+mod] for mod in modules]
    for module in modules:
        mh = hash_module(module)
        modules_hash.append(mh)
    
    watchlist_hash = hash_watchlist(obj, WATCHLIST)
    living_object = hash_methodes_vivantes(obj) 
    dependances = hash_dependances_modules(obj, modules)
    
    payload_parts = modules_hash.copy()
    payload_parts.append(watchlist_hash)
    payload_parts.append(living_object)
    payload_parts.append(dependances)
    
    # Combinaison
    payload = "|".join(payload_parts)
    full_hash = hashlib.sha256(payload.encode()).hexdigest()

    return {
        "source_hash": modules_hash,
        "watchlist_hash": watchlist_hash,
        "live_object":living_object,
        "dependances":dependances,
        "full_hash": full_hash
    }

def get_full_object_hash(obj, modules=[], WATCHLIST=[]):
    """
    Returns a full_hash, combining all hashes
    - "full_hash": full_hash
    """
    
    modules_hash = []
    package = obj.__module__.split('.', 1)[0]
    modules = [sys.modules[package+"."+mod] for mod in modules]
    """for module in modules:
        mh = hash_module(module)
        modules_hash.append(mh)"""
    
    watchlist_hash = hash_watchlist(obj, WATCHLIST)
    living_object = hash_methodes_vivantes(obj) 
    #dependances = hash_dependances_modules(obj, modules)
    
    payload_parts = []
    #payload_parts = modules_hash.copy()
    payload_parts.append(watchlist_hash)
    payload_parts.append(living_object)
    #payload_parts.append(dependances)
    
    # Combinaison
    payload = "|".join(payload_parts)
    full_hash = hashlib.sha256(payload.encode()).hexdigest()

    return full_hash

# examples:
# get_big_integrity_hash(quiz, modules=["main", "utils"])
# get_full_object_hash(quiz_dev, modules=["main", "utils"], WATCHLIST=['retries'])


def prep(n, quiz_id, quizfile, internetOK=False, p=None):
    if p is None: p=n 
    if n > 0:
        h = prep(n - 1, quiz_id, quizfile, internetOK, p)
    else:
        stack = inspect.stack() 
        context = [frame.function for frame in stack[2:7]]
        nc = None
        frame = inspect.stack()[p+2].frame

        if 'self' in frame.f_locals:
            nc = frame.f_locals['self'].__class__.__name__
        elif 'cls' in frame.f_locals:
            nc = frame.f_locals['cls'].__name__
        context.extend([nc, quizfile])
        while len(context) < 6: context.append("none")
        context.extend([internetOK, quiz_id])

        fingerprint = "-".join(context).encode()
        hash_digest = hashlib.sha256(fingerprint).digest()
        h = base64.urlsafe_b64encode(hash_digest)
    return h


def is_encrypted(s):
    from cryptography.fernet import Fernet, InvalidToken
    h = b'zzFJ0WBRXuFC1qJOFGxeqt3NUNcFR9vdZqFkCry-DYw='
    f = Fernet(h)
    try:
        dec = f.decrypt(s).decode('utf-8')
        return True
    except: # InvalidToken:
        return False

def calculate_quiz_score(quiz_type, user_answers, propositions, weights=None, constraints=None):
    """ 
    constraints: List of dicts ex: [{"indices": (0, 1), "type": "XOR", "malus": 2}] 
    - XOR (Exclusion) A and B must be different 
    - IMPLY (Implication) If A is TRUE then B is TRUE 
    - SAME (Coherence) A and B are equivalent (same value) 
    - IMPLYFALSE: If A is TRUE, then B MUST be FALSE 
    """ 
    # Reference matrix (If nothing is defined in the proposal) and if 
    # another reference matrix is not passed as an argument 
    # Format: (User_Response, Expected)
    
    if weights is None:
        default_weights = {
                (True, True):   1,  # Vrai Positif
                (True, False): -1,  # Faux Positif 
                (False, True):  0,  # Faux Négatif (oubli)
                (False, False): 0   # Vrai Négatif
            }
    else:
        default_weights = weights

    TYPE_MAP = {
        "int": int,
        "bool": lambda s: s.lower() == "true" if isinstance(s, str) else s,
        "float": float,
               }
    
    score = 0.0
    total_possible = 0.0
    if not isinstance(user_answers, dict): 
        print(f"❌ Erreur : user_answers doit être un dictionnaire")

    # Correction in the case of a template
    if quiz_type in ['numeric-template', 'mcq-template']:
        context = user_answers.get('context', {})
        #print(context)
        for p in propositions:
            pexpect =  p["expected"]
            ptype = p.get("type", bool if quiz_type == "mcq" else float)
            ptype = TYPE_MAP.get(ptype, ptype)
            if "modules" in context:  #we import the modules necessary to calculate the answer - HOT
                toexec = ""
                for k,v in context["modules"].items():
                    toexec += f"import {v} as {k}\n"
                toexec += "result="+pexpect
                exec(toexec, context)
                p["expected"] = ptype(context["result"])
            else:    
                p["expected"] = ptype(eval(pexpect,{},context)) if isinstance(pexpect, str) else pexpect
            p['proposition'] = p['proposition'].format(**context)
        quiz_type = quiz_type.split('-')[0]
        user_answers.pop("context", None)
        
    for answer, prop in zip(user_answers.values(), propositions):
        expected = prop.get("expected", None)
        # default was False
        if expected is None: return 0, 1 #No solution known in proposition
        user_val = bool(answer)
        case = (user_val, expected)
        
        if quiz_type == "mcq":
            #1. Calculation of the theoretical total
            total_possible += prop.get("bonus", default_weights[(True, True)]) if expected \
                   else prop.get("bonus", default_weights[(False, False)])

            #2. Calculation of the score for this proposition
            if user_val == expected:
                # Correct Case (VP ou VN)
                val = prop.get("bonus", default_weights[case])
                score += val
            else:
                # Incorrect Case (FP or FN)
                val = prop.get("malus", default_weights[case])              
                # We make sure that the penalty is indeed deducted
                score -= abs(val)

        elif quiz_type == "numeric":
            bonus = prop.get("bonus", 1)
            total_possible += bonus
            diff = abs(answer - prop.get("expected", 0))
            tol = max(prop.get("tolerance_abs", 0), 
                      prop.get("tolerance", 0.01) * abs(prop.get("expected", 0))) 
            score += bonus if diff <= tol else -abs(prop.get("malus", 0))

           
    # contraints
    if quiz_type == "mcq" and constraints:
        for rule in constraints:
            idx1, idx2 = rule["indices"]
            ans1, ans2 = bool(user_answers[idx1]), bool(user_answers[idx2])

            violation = False
            r_type = rule.get("type", "XOR").upper()

            if r_type == "XOR": #The two cannot be identical
                if not (ans1 != ans2) :
                    violation = True
                    #print(f"violation XOR entre {idx1} et {idx2}")
            elif r_type == "IMPLY": # If 1 is TRUE, then 2 MUST be TRUE (otherwise contradiction)
                if ans1 and not ans2:
                    violation = True
                    #print(f"violation IMPLY entre {idx1} et {idx2}")
            elif r_type == "IMPLYFALSE": # If 1 is TRUE, then 2 MUST be FALSE (otherwise contradiction)
                if ans1: 
                    if ans2:
                        violation = True
                    #print(f"violation IMPLY entre {idx1} et {idx2}")
            elif r_type == "SAME": # Must have the same value
                if ans1 != ans2:
                    violation = True
                    #print(f"violation SAME between {idx1} and {idx2}")

            if violation:
                # We subtract the contradiction malus
                score -= abs(rule.get("malus", 1))

    return score, total_possible



def get_package_directory(package_name):
    """ Returns the directory containing the package whose name is given.

    Settings
    ----------
    package_name: str
        Package name

    Returns
    -------
    Path
        Package folder

    Raises
    -------
    ImportError
        If the package is not available.

    # Example of use
    package_dir = get_package_directory("labquiz")
    print(package_dir)  
    """
    import importlib.util
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        raise ImportError(f"Package {package_name} is unavailable.")
    return Path(spec.origin).parent

def get_package_hash(package_name):
    """ Returns the hash of the package whose name is passed.
    Settings
    ----------
    package_name: str
        Package name

    Returns
    -------
    str
        Package hash (or None if the package has no hash)

    Raises
    -------
    ImportError
        If the package is not available.

    Notes
    -----
    The hash is stored in the Keywords field of the METADATA file
    of the package. It starts with "hash:".

    Example of use
    ---------------------
    package_hash = get_package_hash("labquiz")
    print(package_hash)
    """
    from importlib.metadata import metadata
    meta = metadata(package_name)
    #print(meta)
    keywords = meta.get_all('Keywords', [])
    # We are looking for the element that begins with "hash:"
    kz = keywords[0]
    for k in kz.split(','):
        if k.startswith("hash:"):
            return k.split(":", 1)[1]
    return None

def package_hash(package_dir, exclude=None, algo="sha256"):
    """
    Compute the hash of a package.

    Parameters
    ----------
    package_dir : Path or str
        Path to the package directory.
    exclude : set or list, optional
        List of file names or directory names to exclude from the hash computation.
    algo : str, optional
        Hash algorithm to use (default is 'sha256').

    Returns
    -------
    tuple
        A tuple containing the hash as a string and the list of file paths that were used to compute the hash.
    """
    
    package_dir = Path(package_dir) if isinstance(package_dir, str) else package_dir
    h = hashlib.new(algo)
    exclude = exclude or set()

    files = sorted(
        p for p in package_dir.rglob("*")
        if p.is_file() and p.name not in exclude and  not any(ex in p.parts for ex in exclude)
    )

    for path in files:
        h.update(path.read_bytes())

    return h.hexdigest(), files

def check_installed_package_integrity():
    from importlib.metadata import version, PackageNotFoundError
    import importlib.util
    
    spec = importlib.util.find_spec("labquiz")
    if spec is None:
        print("⚠️ LabQuiz is not installed!")
        return 

    PACKAGE_NAME = __package__.split(".", 1)[0]
    try:
        __version__ = version(PACKAGE_NAME)
    except PackageNotFoundError:
        try:
            __version__  =  version("labquiz") + " (dev)"
        except PackageNotFoundError:
            return "non installé"
        
    EXCLUDE = {"putils.py", "__pycache__", ".ipynb_checkpoints", ".DS_Store"}
    labdir = get_package_directory("labquiz")
    installed_hash, f = package_hash(labdir, exclude=EXCLUDE)
    recorded_hash = get_package_hash("labquiz")
    if installed_hash != recorded_hash:
        print(f"""⚠️ Package hash different than expected -- this is recorded 
              Installed: {installed_hash} 
              Expected: {recorded_hash}""")
        return False
    print(f"LabQuiz, {PACKAGE_NAME} version {__version__}")
    return True
#_ = check_installed_package_integrity()

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _fernet_key_from_password(password: str) -> bytes:
    return base64.urlsafe_b64encode(
        hashlib.sha256(password.encode()).digest()
    )

def compute_local_hash(base_dir: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(base_dir.rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()


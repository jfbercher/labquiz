import base64, sys
import hashlib, inspect, re, types
from importlib.metadata import metadata
from pathlib import Path
import requests, json, datetime
import asyncio


def getUser():
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


def get_source_integrity_hash(cls):
    """
    Calcule un hash basé sur le texte source des méthodes de la classe.
    """
    try:
        # 1. Code source de la classe entière
        source = inspect.getsource(cls)
        
        # 2. Nettoyage pour plus de stabilité : suppression des commentaires, lignes vides

        source = re.sub(r'#.*', '', source) 
        source = "\n".join([line.strip() for line in source.splitlines() if line.strip()])
        
        # 3. Hashage
        return hashlib.sha256(source.encode('utf-8')).hexdigest()
    
    except (OSError, TypeError):
            return "Source non disponible"



def get_ultra_integrity_hash(cls):
    """
    Calcule un hash basé sur le bytecode ET les constantes (valeurs fixes)
    des méthodes définies localement dans la classe.
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
            
            # 1. Le bytecode (la logique)
            bytecode = code_obj.co_code
            
            # 2. Les constantes (les valeurs hardcodées : strings, int, etc.)
            # On convertit en string pour pouvoir le hasher facilement
            consts = str(code_obj.co_consts).encode('utf-8')
            
            # On combine le nom, le bytecode et les constantes
            signature = f"{name}:".encode('utf-8') + bytecode + consts
            method_hashes.append(hashlib.sha256(signature).hexdigest())

    if not method_hashes:
        return None

    method_hashes.sort()
    final_payload = "".join(method_hashes).encode('utf-8')
    return hashlib.sha256(final_payload).hexdigest()




def get_package_directory(package_name):
    """
    Retourne le répertoire contenant le package dont le nom est donné.

    Parameters
    ----------
    package_name : str
        Nom du package

    Returns
    -------
    Path
        Dossier du package

    Raises
    -------
    ImportError
        Si le package n'est pas disponible.

    # Exemple d'utilisation
    package_dir = get_package_directory("labquiz")
    print(package_dir)  
    """
    import importlib.util
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        raise ImportError(f"Le package {package_name} n'est pas disponible.")
    return Path(spec.origin).parent

def get_package_hash(package_name):
    """
    Retourne le hash du package dont le nom est passé.

    Parameters
    ----------
    package_name : str
        Nom du package

    Returns
    -------
    str
        Hash du package (ou None si le package n'a pas de hash)

    Raises
    -------
    ImportError
        Si le package n'est pas disponible.

    Notes
    -----
    Le hash est stocké dans le champ Keywords du fichier METADATA
    du package. Il commence par "hash:".

    Exemple d'utilisation
    ---------------------
    package_hash = get_package_hash("labquiz")
    print(package_hash)
    """
    from importlib.metadata import metadata
    meta = metadata(package_name)
    #print(meta)
    keywords = meta.get_all('Keywords', [])
    # On cherche l'élément qui commence par "hash:"
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

def compute_local_hash(base_dir: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(base_dir.rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()

# -------------- Update do_the_check -----------------------------


####### Encore une nouvelle version des Hash ############
# indépendant de la version de python
# décomposé en petits modules
# - code source initial
# - objet vivant
# - watchlist
# dépendances utilisées

## Avec parcours des sources du module
## bytecodes supprimés


# ---------- Fonctions de base ----------

def hash_fonction(func: callable) -> bytes:
    """
    Hash robuste d'une fonction ou méthode.
    """
    h = hashlib.sha256()

    # Source si disponible
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

    # Cas méthode liée → récupérer la fonction réelle
    func = getattr(func, "__func__", func)

    # Fonction Python pure
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

        # localisation runtime (IPython s’en sert)
        h.update(code.co_filename.encode())
        h.update(str(code.co_firstlineno).encode())

    else:
        # builtin / C-extension
        h.update(repr(func).encode())

    return h.digest()


# ---------- Fonctions de premier niveau ----------

def fonctions_premier_niveau(module: ModuleType) -> Iterable[Tuple[str, callable]]:
    for name, obj in vars(module).items():
        if inspect.isfunction(obj) and (obj.__module__ == module.__name__ or obj.__module__ == '__main__'):
            yield name, obj


# ---------- Classes ----------

def methodes_de_classe_old(cls) -> Iterable[Tuple[str, callable]]:
    """
    Retourne les méthodes définies directement dans la classe (pas héritées).
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
    Retourne les méthodes définies directement dans la classe (pas héritées, pas importées).
    """
    for name, obj in cls.__dict__.items():

        # unwrap staticmethod / classmethod
        if isinstance(obj, (staticmethod, classmethod)):
            func = obj.__func__
        elif inspect.isfunction(obj):
            func = obj
        else:
            continue

        # FILTRE CRUCIAL : définie lexicalement dans la classe
        if not func.__qualname__.startswith(cls.__name__ + "."):
            continue

        yield name, func


def methodes_de_classe_plus(cls):
    """
    Retourne les méthodes définies directement dans la classe
    (pas héritées, pas importées).
    """
    for name, obj in cls.__dict__.items():

        # 1. Extraire l'objet fonction réel
        if isinstance(obj, (staticmethod, classmethod)):
            func = obj.__func__
        elif inspect.isfunction(obj):
            func = obj
        else:
            continue

        # 2. Éliminer les fonctions venant d'un autre module
        if func.__module__ != cls.__module__:
            continue

        # 3. Éliminer les fonctions non définies lexicalement dans la classe
        if not func.__qualname__.startswith(cls.__name__ + "."):
            continue
        
        yield name, func


def hash_classe(cls) -> bytes:
    """
    Hash d'une classe basée sur ses méthodes et son nom.
    """
    h = hashlib.sha256()

    # Nom + bases
    h.update(cls.__name__.encode("utf-8"))
    for base in cls.__bases__:
        h.update(base.__name__.encode("utf-8"))

    # Méthodes
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

## C'est celui qui marche !

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

        # méthode liée → fonction réelle
        func = getattr(attr, "__func__", attr)

        # 1. Exclure imports externes
        if func.__module__ != module_name:
            continue

        # 2. Exclure méthodes héritées externes
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
    Hash des fonctions globales dans les modules_cibles utilisées par les méthodes de la classe.
    
    obj : classe dont on veut sécuriser le graphe
    modules_cibles : liste de noms de modules à inclure (ex: ["main", "utils"])
    """
    
    cls = obj.__class__
    module_name = cls.__module__
    package = obj.__module__.split('.', 1)[0]
    
    h = hashlib.sha256()
    # extraire les dépendances utilisées par les méthodes de la classe
    dependances = set()
    for name, func in methodes_vivantes_objet(obj):
        dependances |= set(func.__code__.co_names)
    #print(dependances)
    # parcourir les modules ciblés seulement
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
                # si besoin, on peut aussi hasher d'autres objets, par exemple classes ou constantes
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
    Retourne un dict avec :
    - "source_hash": modules_hash,      # Les sources
    - "watchlist_hash": watchlist_hash, # les paramètres surveillés
    - "live_object":living_object,      # les méthodes de l'objet vivant -- identifie monkey patching sur l'objet
    - "dependances":dependances,       # les méthodes vivantes des modules utilisées par l'objet courant
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
    Retourne un dict avec :
    - "source_hash": modules_hash,      # Les sources
    - "watchlist_hash": watchlist_hash, # les paramètres surveillés
    - "live_object":living_object,      # les méthodes de l'objet vivant -- identifie monkey patching sur l'objet
    - "dependances":dependances,       # les méthodes vivantes des modules utilisées par l'objet courant
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

# exemples:
# get_big_integrity_hash(quiz, modules=["main", "utils"])
# get_full_object_hash(quiz_dev, modules=["main", "utils"], WATCHLIST=['retries'])


async def do_the_check(obj, WORKDIR):
    import shutil
    
    if not obj.stop_event.is_set(): obj.stop_event.set() #Arrêt du premier daemon
    session_hash = compute_local_hash(WORKDIR)  #Vérification pas de modif dans WORKDIR
    shutil.rmtree(WORKDIR)
    
    while not obj.stop_check_event.is_set():
        parameters = {
                "exam_mode": obj.exam_mode,
                "test_mode": obj.test_mode,
                "retries": obj.retries,
                "counts": ",".join(str(v) for k, v in sorted(obj.quiz_counts.items())),
                "corrections": ",".join(str(v) for k, v in sorted(obj.quiz_correct.items())),
                "full_hash": get_full_object_hash(obj, modules = ['main', 'utils'],
                         WATCHLIST=['exam_mode', 'test_mode', 'retries'])
               }
        
        big_hash = get_big_integrity_hash(obj, modules = ['main', 'utils'],
                                 WATCHLIST=['exam_mode', 'test_mode', 'retries'])
        
        parameters['get_big_integrity_hash'] = big_hash
        parameters['session_hash'] = session_hash
        
        payload = {
        "notebook_id": obj.machine_id,
        "student": obj.student.name,
        "quiz_title": "integrity",
        "timestamp": datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"), #datetime.datetime.now().isoformat(timespec="seconds"),
        "event_type": "z_check_integrity",    
        "parameters": parameters,
        "answers": {},                
        "score": str(0)
        }
        

        try:
            #requests.post(self.SHEET_URL, json=payload)
            r = requests.post(
            obj.SHEET_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "text/plain"}
            )
        except Exception as e:
            print("⚠️ Erreur d’envoi :", e) 
            
        try:
            # Attend 600s OU que stop_event soit activé
            await asyncio.wait_for(obj.stop_check_event.wait(), timeout=obj._CHECKALIVE)
        except asyncio.TimeoutError:
            # Si le timeout expire, on continue simplement la boucle while
            pass
        
        
# --------- Fin update do the check ---------------------        
        
def do_the_check_old(obj, WORKDIR):
    session_hash = compute_local_hash(WORKDIR)
    #print("SESSION_HASH:", session_hash)

    EXCLUDE = {"putils.py", "__pycache__", ".ipynb_checkpoints", ".DS_Store"}
    labdir = get_package_directory("labquiz")
    installed_hash, f = package_hash(labdir, exclude=EXCLUDE)
    output = {
    'installed_hash' : installed_hash,
    'session_hash': session_hash,
    'full_hash': get_full_object_hash(obj),
    'src_hash': get_source_integrity_hash(obj.__class__),
    'retries': obj.retries,
    'exam_mode': obj.exam_mode, 
    'test_mode': obj.test_mode,
    'transfer': obj.sheetTransfer,
    'quizfile': obj.QUIZFILE
    }

    payload = {
    "notebook_id": obj.machine_id,
    "student": obj.student.name,
    "quiz_title": "integrity",
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "event_type": "teacher_check",    
    "answers": output,                
    "score": 0
    }

    try:
        #requests.post(self.SHEET_URL, json=payload)
        r = requests.post(
        obj.SHEET_URL,
        data=json.dumps(payload),
        headers={"Content-Type": "text/plain"}
        )
    except Exception as e:
        print("⚠️ Erreur d’envoi :", e)
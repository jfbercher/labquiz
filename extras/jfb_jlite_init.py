"""import sys
if sys.platform == "emscripten":
    import micropip
    import js
"""    
def get_base_url():
    import js
    # On récupère l'URL du worker 
    worker_url = js.globalThis.location.href
    
    try:
        # Tentative de récupérer l'URL de la fenêtre principale
        raw_url = js.globalThis.parent.location.href
    except:
        # Si restriction de sécurité, on nettoie l'URL du worker
        raw_url = worker_url.split('/extensions/')[0]

    # On nettoie pour enlever /lab ou /retro et les paramètres ?path=...
    base = raw_url.split('?')[0].split('/lab/')[0].split('/retro/')[0].split('/index.html')[0]
    
    # Nettoyage final pour s'assurer qu'il n'y a pas de slash traînant
    if base.endswith('/'):
        base = base[:-1]
    return base


async def setup_env(base_url, toinstall): 
    import micropip
    for item in toinstall:  
        try:
            await micropip.install(item)
            print(f"✅ {item} installé")
        except Exception as e:
            print(f"❌ Erreur : {e}")

    try:
        import pyodide_http
        pyodide_http.patch_all()
        print("✅ Patch pyodide-http activé")
        import requests
        print("✅ Requests importé")
    except Exception as e:
        print(f"❌ Erreur : {e}")

async def update_files(toinstall):
    # Certains types sont mal reconnus. Ils sont installés avec une double extension, 
    # la seconde extension permettant de fixer le type. On les renomme ensuite. 
    import os
    ll = os.listdir()
    dbl_extensions = [f for f in ll if f.count('.') >= 2]
    for file in dbl_extensions:
        newfilename = file.rsplit('.',1)[0]
        if not os.path.isfile(newfilename): 
            os.rename(file, newfilename)
            print(f"✅ {file} mis à jour en {newfilename}")  
        else: 
            newfilename = newfilename.rsplit('.',1)[0] + '_new.'+ newfilename.rsplit('.',1)[1]
            os.rename(file, newfilename)
            print(f"✅ {file} mis à jour en {newfilename}")  

"""
    # ça marche !!
    from pyodide.http import pyfetch
    import os
"""

async def sync_simple(file_list):
    base_url = 'https://perso.esiee.fr/~bercherj/jlite/'
    base_url = get_base_url()
    for file_path in file_list:
        try:
            # pyfetch gère mieux les chemins relatifs au site
            full_url = base_url + '/' + file_path
            response = await pyfetch(full_url)
            data = await response.bytes()

            # Créer le dossier local
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            with open(file_path, "wb") as f:
                f.write(data)
            print(f"✅ OK : {file_path}")
            
        except Exception as e:
            print(f"💥 Échec critique sur {file_path} : {e}")
"""
else:
    print("⚠️ mauvais environnement (jupyterlite/pyodide requis)")
"""
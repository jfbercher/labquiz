# zutils.py doit être présent dans le répertoire local
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
    password_open="mot_pour_ouvrir",
    password_seal="secret_enseignant23"
)

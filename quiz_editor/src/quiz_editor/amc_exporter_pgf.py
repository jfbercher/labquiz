import re
import random

# --- CONFIGURATION DU TEMPLATE LATEX ---
LATEX_TEMPLATE = r"""\documentclass[a4paper]{article}

\usepackage[utf8x]{inputenc}    
\usepackage[T1]{fontenc}
\usepackage[french]{babel}
\usepackage{pgfmath}
\usepackage[automult,ensemble]{automultiplechoice}    

\begin{document}

% Préparation du groupe de questions pour le mélange
\shufflegroup{tout}

\onecopy{10}{    

%%% En-tête des copies %%%
\noindent{\bf QCM de Mathématiques \hfill Examen Dynamique}

\vspace*{.5cm}
\begin{center}\em
Les questions avec le symbole \multiSymbole{} peuvent avoir plusieurs bonnes réponses.
Les calculs sont générés aléatoirement pour chaque copie.
\end{center}
\vspace*{1ex}

%%% Insertion des questions %%%
\insertgroup{tout}

\clearpage
}

\end{document}
"""

def python_expr_to_pgf(expr):
    """Traduit une expression Python simple en syntaxe PGFMath."""
    expr = expr.replace('math.', '').strip()
    # Nettoyage des f-strings résiduelles f'{...}'
    expr = re.sub(r"^[fF]?['\"]+\{(.*?)\}.*$", r"\1", expr)
    # Remplacement des variables {a} par (\mya)
    expr = re.sub(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}', r'(\\my\1)', expr)
    # Opérateurs
    expr = expr.replace('**', '^')
    expr = expr.replace('log(', 'ln(')
    return expr

def convert_to_amc_latex(data, use_negative_points=True):
    """Génère un fichier .tex complet compatible AMC avec PGFMath."""
    questions_code = []
    
    # On itère sur les questions (triées par quiz1, quiz2...)
    for q_id, q_content in data.items():
        if q_id == "title" or not isinstance(q_content, dict):
            continue
        
        q_type = str(q_content.get('type', '')).lower()
        is_template = q_type.endswith('-template')
        props = q_content.get('propositions', [])
        
        # Type de question AMC
        is_mult = "multiple" in q_type or len([p for p in props if p.get('expected') is True]) > 1
        amc_tag = "questionmult" if is_mult else "question"
        q_label = q_content.get('label', f'q:{q_id}')
        q_category = q_content.get('category', 'nocategory')

        # Construction du bloc question
        block = [f"\\element{{{q_category}}}{{"]
        block.append(f"  \\begin{{{amc_tag}}}{{{q_label}}}")
        
        # 1. Génération des variables (uniquement si template)
        if is_template:
            raw_text = str(q_content.get('question', '')) + str(props)
            found_vars = set(re.findall(r'\{([a-zA-Z_][a-zA-Z0-9_]*)[\}:]', raw_text))
            
            params = q_content.get('parameters', {})
            for var in found_vars:
                p = params.get(var, {'min': 1, 'max': 10, 'step': 1})
                v_min, v_max, v_step = p.get('min', 1), p.get('max', 10), p.get('step', 1)
                # PGFMath : génération aléatoire respectant le pas (step)
                block.append(f"    \\pgfmathsetmacro{{\\my{var}}}{{round((rnd*({v_max}-{v_min})+{v_min})/{v_step})*{v_step}}}")

        # 2. Texte de la question
        q_text = q_content.get('question', '')
        q_text_pgf = re.sub(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}', r'\\my\1', q_text)
        block.append(f"    {q_text_pgf}")

        # 3. Réponses
        if "numeric" in q_type:
            for i, p in enumerate(props):
                pgf_expr = python_expr_to_pgf(p.get('expected', '0'))
                block.append(f"    \\pgfmathsetmacro{{\\res{i}}}{{{pgf_expr}}}")
                block.append(f"    \\AMCNumericChoices{{\\res{i}}}{{digits=3,decimals=2,sign=true}}")
        else:
            block.append("    \\begin{choices}")
            for p in props:
                is_correct = p.get('expected') is True
                cmd = "\\correctchoice" if is_correct else "\\wrongchoice"
                p_text = re.sub(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}', r'\\my\1', p.get('label', p.get('proposition', '')))
                
                bonus = 1 if is_correct else 0
                malus = -1 if (use_negative_points and not is_correct) else 0
                block.append(f"      {cmd}{{{p_text}}} \\scoring{{b={bonus},m={malus}}}")
            block.append("    \\end{choices}")

        block.append(f"  \\end{{{amc_tag}}}")
        block.append(f"}}")
        
        questions_code.append("\n".join(block))
    
    # Assemblage final dans le template
    return LATEX_TEMPLATE.replace("%%% Insertion des questions %%%", "\n".join(questions_code))
import numpy as np
import pandas as pd

from i18n import get_translator
from convert_utils import evaluate_fstring, evaluate_text, safe_eval, processPropositions


def convert_to_amc_latex(data, use_negative_points=True):
    latex_output = []
    
    
    for q_id, q_content in data.items():
        if q_id == "title" or not isinstance(q_content, dict): continue
        
        q_type = str(q_content.get('type', '')).lower()
        # RÈGLE : Uniquement les types -template passent en mode dynamique
        is_template = 'template' in q_type
        
        context = {}
        if is_template:
            latex_output.append(f"\n% --- {q_id} is a template - Generating random values ---")
            variables = q_content.get('variables', [])
            for var_name in variables.keys():
                engine = variables[var_name].get('engine')
                engine_call = variables[var_name].get('call')
                engine_prefix = "rng." if engine == "numpy rng." else "pd."
                expression = f"{engine_prefix}{engine_call}"
                context[var_name] = safe_eval(expression)
                latex_output.append(f"% {var_name} = {context[var_name]}")
        else:
            latex_output.append(f"\n% --- {q_id} ---")
        

        # Evaluation (won't do anything if context is empty, so respect static text)

        q_text = evaluate_fstring(q_content.get('question', ''), context)
        q_category = q_content.get('category', 'nocategory')
        q_label = q_content.get('label', f'q:{q_id}')
        
        # Détection question / questionmult
        props = q_content.get('propositions', [])
        is_mult = "multiple" in q_type or len([p for p in props if p.get('expected') is True]) > 1
        amc_tag = "questionmult" if is_mult else "question"

                # Construction du bloc question
        latex_output.append(f"\\element{{{q_category}}}{{")
        latex_output.append(f"  \\begin{{{amc_tag}}}{{{q_label}}}")
        latex_output.append(f"  {q_text}")

        if "numeric" in q_type:
            for p in props:
                v_prop, val_eval, v_rep, q_label = processPropositions(p, q_type, context)
                latex_output.append(f"  \\AMCNumericChoices{{{val_eval}}}{{digits=3,decimals=2,sign=true}}")
        else:
            latex_output.append("    \\begin{choices}")
            for p in props:
                v_prop, v_exp, v_rep, q_label = processPropositions(p, q_type, context)
                is_correct = v_exp
                cmd = "\\correctchoice" if is_correct else "\\wrongchoice"
                
                # Scoring logique
                bonus = 1 if is_correct else 0
                malus = -1 if (use_negative_points and not is_correct) else 0
                latex_output.append(f"      {cmd}{{{v_prop}}} \\scoring{{b={bonus},m={malus}}}")
            latex_output.append("    \\end{choices}")

        latex_output.append(f"\\end{{{amc_tag}}}")
        latex_output.append(f"}}")
            
    return "\n".join(latex_output)
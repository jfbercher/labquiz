import re
import numpy as np
import pandas as pd
import html
import math
import html

rng = np.random.default_rng()

def strip_f_prefix(template: str) -> str:
    """Removes the 'f' prefix from the template string."""
    import re
    return re.sub(r'^\s*f([\'"]{1,3})', r'\1', template, count=1)

def evaluate_fstring_previous(template, context):
    if not isinstance(template, str): return template
    if not '{' in template: return template

    template = strip_f_prefix(template)

    # Replace $.{...}..$ with {{...}}
    template = re.sub(
        r'(?<!\\)\$(.+?)(?<!\\)\$',
        lambda m: '$' + m.group(1).replace('{', '{{').replace('}', '}}') + '$',
        template,
        flags=re.DOTALL
    )

    safe_globals = {
        "__builtins__": {},
        "np": np,
        "math": math,
    }
    #print("safe_globals", safe_globals)
    #print("template", template)
    #print("context", context)
    val = eval("f" + repr(template), safe_globals, context).strip("'").strip('"')
    #print("val", val)
    #if isinstance(val, bool): val = str(val).lower()
    return val

def evaluate_fstring(template, context):
    import re
    if not isinstance(template, str): return template
    template = strip_f_prefix(template)
    
    # Replace $.{...}..$ with {{...}} for latex commands but not for possible "f-strings"
    template = re.sub(
    r'(?<!\\)\$(.+?)(?<!\\)\$',
    lambda m: '$' + re.sub(
        r'\\[a-zA-Z]+\{[^{}]*\}',
        lambda c: c.group(0).replace('{', '{{').replace('}', '}}'),
        m.group(1)
    ) + '$',
    template,
    flags=re.DOTALL
    )
    
    #template = replace_latex_braces(template)

    safe_globals = {
        "__builtins__": {},
        "np": np,
    }

    
    val = eval("f" + repr(template), safe_globals, context).strip("'").strip('"')
    return val

def safe_eval(expr):
    """
    Evaluate expression in a restricted namespace.
    Only rng, numpy and pandas are allowed.
    """
    return eval(expr, {"__builtins__": {}}, {"rng": rng, "np": np, "pd": pd})

def evaluate_text(text, context):
    return html.escape(evaluate_fstring(text, context)).strip("'").strip('"')

def processPropositions(p, q_type, context):
    v_exp = p.get('expected', '') 
    v_prop = p.get('proposition', '')
    v_rep = p.get('answer', '')
    v_lab = evaluate_text(p.get('label', ''), context)
    if 'template' in q_type:
        if not '{' in v_exp: v_exp = f'{{ {v_exp} }}'
        v_exp = evaluate_fstring(v_exp, context)
        v_exp = v_exp.strip().strip("'").strip('"')
        v_prop = evaluate_text(v_prop, context)
        v_rep = evaluate_text(v_rep, context)
        if "numeric" in q_type:
            v_exp = float(v_exp) # to extend later with type checking
        else:
            v_exp = v_exp == 'True'
    return v_prop, v_exp, v_rep, v_lab

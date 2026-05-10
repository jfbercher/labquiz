import re
import json
import random
import html
import numpy as np
import pandas as pd
from i18n import get_translator
from convert_utils import evaluate_text, safe_eval, processPropositions, looks_like_markdown, to_html

rng = np.random.default_rng()


# ---------------------------------------------------------------------------
# Parse_js_var_spec() utility function
#
# Parse both supported forms of call for template variables
# and returns a spec dict serializable in JSON (consumed by the JS).
# Returns None if the form is not recognized.
# ---------------------------------------------------------------------------
def parse_js_var_spec(engine: str, call: str):
    """
    Recognizes the two patterns that can be generated on the JS side: 
    - integers(min, max, [endpoint,] size={size_expr}) 
    - normal(mean, std, size={size_expr}) 
    Returns a dict {"type", ...fields...} or None if not recognized. 
    size_expr can be a literal integer OR the name of a variable already defined.
    """
    if engine != "numpy rng.":
        return None

    # integers(min, max, …, size=N  ou  size={var})
    m = re.match(
        r'integers\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)'
        r'(?:\s*,\s*[^,)]+)*'          # paramètres optionnels (endpoint…)
        r'(?:\s*,\s*size\s*=\s*\{?(\w+)\}?)?\s*\)',  # size= optionnel
        call
    )
    if m:
        return {
            "type": "integers",
            "min": int(float(m.group(1))),
            "max": int(float(m.group(2))),
            "size": m.group(3),          # nom de var ou entier littéral
        }

    # normal(mean, std, …, size=N  ou  size={var})
    m = re.match(
        r'normal\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)'
        r'(?:\s*,\s*[^,)]+)*'
        #r'\s*,\s*size\s*=\s*\{?(\w+)\}?\s*\)',
        r'(?:\s*,\s*size\s*=\s*\{?(\w+)\}?)?\s*\)',  # size= optionnel
        call
    )
    if m:
        return {
            "type": "normal",
            "mean": float(m.group(1)),
            "std": float(m.group(2)),
            "size": m.group(3),
        }

    return None


def convert_to_interactive_html(data, lang='en', max_attempts=3, embedImages=True):
    _ = get_translator(lang)
    html_content = []

    from convert_utils import to_html as _to_html
    def to_html(text):  # pour fixer le paramètre embedImages et ne pas ré-écrire tous les to_html()
        return _to_html(text, embedImages=embedImages, protectMaths=True)


    # --- HEADER & STYLE ---
    html_content.append("""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Quiz Interactif</title>
    <script>window.MathJax = {{ tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] }} }};</script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; max-width: 850px; margin: auto; padding: 20px; background: #f0f4f8; color: #2d3436; }}
        #global-score-banner {{
            position: sticky; top: 0; background: #0984e3; color: white;
            padding: 15px; text-align: center; font-size: 1.4em; font-weight: bold;
            border-radius: 0 0 15px 15px; margin-bottom: 30px; z-index: 1000;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .question-card {{ background: white; padding: 25px; margin-bottom: 30px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); border-left: 6px solid #0984e3; }}
        .question-text {{ font-size: 1.15em; font-weight: 600; margin-bottom: 20px; }}
        .option {{ display: flex; align-items: flex-start; margin: 10px 0; padding: 10px; border-radius: 8px; cursor: pointer; border: 1px solid transparent; }}
        .option:hover {{ background: #f8f9fa; border-color: #eee; }}
        .numeric-input {{ padding: 10px; font-size: 1em; border: 2px solid #dfe6e9; border-radius: 6px; width: 160px; margin: 10px 0; }}
        .explanation-box {{ display: none; margin-top: 10px; margin-bottom: 10px; padding: 15px; border-radius: 8px; background: #e0fbfa; border-left: 4px solid #00cec9; }}
        .btn-group {{ margin-top: 20px; display: flex; gap: 10px; border-top: 1px solid #eee; padding-top: 15px; }}
        .btn {{ padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; transition: 0.2s; }}
        .btn-validate {{ background: #2ecc71; color: white; }}
        .btn-correct {{ background: #00cec9; color: white; }}
        .btn-reset {{ background: #dfe6e9; color: #636e72; }}
        .disabled {{ opacity: 0.4; pointer-events: none; filter: grayscale(1); }}
        .match {{ background: #eafff0 !important; border-color: #2ecc71 !important; }}
        .mismatch {{ background: #fff0f0 !important; border-color: #e74c3c !important; }}
        .attempts-hint {{ font-size: 0.85em; color: #7f8c8d; margin-top: 5px; }}
        table {{
            width: 80%;
            table-layout: fixed;
            margin: auto;
        }}
        td, th {{
            border: 1px solid black;
            padding: 8px;
            text-align: center;
        }}
        table.styled {{
            width: 60%; margin: 1em auto; border-collapse: collapse;
            table-layout: fixed;
            font: 0.9rem -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
            border: 1px solid #e5e7eb; border-radius: 6px; overflow: hidden;
        }}
        table.styled th, table.styled td {{
            padding: 6px 8px; text-align: center;
        }}
        table.styled thead th {{
            background: #f6f8fa; border-bottom: 2px solid #d0d7de; font-weight: 600;
        }}
        table.styled tbody td {{
            border-bottom: 1px solid #d0d7de;
        }}
        table.styled td:first-child,
        table.styled th:first-child {{
            font-weight: 600;
            width: 25%; 
            border-right: 2px solid #d0d7de;            
            white-space: nowrap;
            background: #f9fafb;
        }}
    </style>
</head>
<body>
    <div id="global-score-banner">
        {data_title} <br>
        👉🏼 {global_score} <span id="total-score">0.00</span> / <span id="total-max">0</span>
    </div>
""".format(lang=lang,
           global_score=_("Global Score"),
           data_title=data.get('title', ''),
           )
    )

    for q_id, q_content in data.items():
        if q_id == "title" or not isinstance(q_content, dict):
            continue

        # 1. CONTEXT GENERATION (FOR TEMPLATES)
        context = {}
        q_type = str(q_content.get('type', 'qcm')).lower()

        # Determining the rendering mode of template variables. 
        # - For each variable, we try parse_js_var_spec(). 
        # - If ALL variables are recognized → JS mode (js_mode=True). 
        # - If at least one is not, or engine/call not recognized → complete Python fallback. 
        # js_var_specs: dict {var_name: spec_dict} passed to HTML via data-variables.
        js_mode = False
        js_var_specs = {}

        if "template" in q_type:
            variables = q_content.get('variables', {})
            all_js = True
            for var_name, var_def in variables.items():
                engine = var_def.get('engine', '')
                call   = var_def.get('call', '')
                spec   = parse_js_var_spec(engine, call)
                if spec is None:
                    all_js = False
                    break
                js_var_specs[var_name] = spec

            if all_js and js_var_specs:
                js_mode = True
            else:
                # Fallback : calcul Python
                js_var_specs = {}
                for var_name, var_def in variables.items():
                    engine       = var_def.get('engine', '')
                    engine_call  = var_def.get('call', '')
                    engine_prefix = "rng." if engine == "numpy rng." else "pd."
                    expression   = f"{engine_prefix}{engine_call}"
                    context[var_name] = safe_eval(expression)

        # 2. START CARD
        # In JS mode, we add data-variables (JSON specs) and data-js-template 
        # on card. In non-JS mode, behavior identical to the original.
        if js_mode:
            # We store the raw templates (not evaluated) for JS re-evaluation at Reset
            q_text_raw = q_content.get('question', '')
            q_text_raw = to_html(q_text_raw) #to_html(html.escape(q_text_raw))
            card_extra = (
                f" data-js-template='1'"
                f" data-variables='{json.dumps(js_var_specs, ensure_ascii=False)}'"
                f" data-question-template='{_escape_attr(q_text_raw)}'"
            )
        else:
            q_text_raw = None
            card_extra = ""

        html_content.append(
            f"<div class='question-card' id='card_{q_id}' data-type='{q_type}'"
            f" data-attempts='0'{card_extra}>"
        )

        # Question text
        if js_mode:
            # Placeholder replaced by JS on first render; we leave empty and the JS will fill
            html_content.append(f"<div class='question-text' id='qtext_{q_id}'></div>")
        else:
            q_text = evaluate_text(q_content.get('question', ''), context)
            q_text = to_html(q_text)
            html_content.append(f"<div class='question-text'>{q_text}</div>")

        props = q_content.get('propositions', [])
        random.shuffle(props)

        # 3. QUESTION CONTENT
        if "numeric" in q_type:
            html_content.append("<div class='numeric-container'>")
            for i, p in enumerate(props):
                if js_mode:
                    # En mode JS/numeric, on stocke les templates bruts dans des
                    #         data-attributes ; le JS calculera expected, prop et rep.
                    v_prop_tpl  = to_html(p.get('proposition', ''))
                    v_exp_tpl   = p.get('expected', '')
                    if not '{' in v_exp_tpl:
                        v_exp_tpl = f'{{ {v_exp_tpl} }}'
                    v_rep_tpl   = to_html(p.get('answer', ''))
                    tol_abs     = p.get('tolerance_abs', 0)
                    tol         = p.get('tolerance', 0.01)
                    html_content.append("""
                <div class='numeric-unit'
                     data-prop-template='{v_prop_tpl}'
                     data-exp-template='{v_exp_tpl}'
                     data-rep-template='{v_rep_tpl}'>
                    <label id='prop_{q_id}_{i}'></label><br>
                    <input type='number' step='any' class='numeric-input' id='input_{q_id}_{i}'
                           data-expected='' data-tol-abs='{tol_abs}' data-tol-rel='{tol}'>
                    <div class='explanation-box' id='expl_{q_id}_{i}'><b>{answer}</b> <span class='exp-val'></span><br><span class='rep-val'></span></div>
                </div>""".format(
                        v_prop_tpl=_escape_attr(v_prop_tpl),
                        v_exp_tpl=_escape_attr(v_exp_tpl),
                        v_rep_tpl=_escape_attr(v_rep_tpl),
                        q_id=q_id, i=i,
                        tol_abs=tol_abs, tol=tol,
                        answer=_("Answer:")
                    ))
                else:
                    v_prop, v_exp, v_rep, v_lab, v_tip = processPropositions(p, q_type, context)
                    html_content.append("""
                <div class='numeric-unit'>
                    <label>{v_prop}</label><br>
                    <input type='number' step='any' class='numeric-input' id='input_{q_id}_{i}'
                           data-expected='{v_exp}' data-tol-abs='{tol_abs}' data-tol-rel='{tol}'>
                    <div class='explanation-box' id='expl_{q_id}_{i}'><b>{answer}</b> {v_exp}<br>{v_rep}</div>
                </div>""".format(
                        v_prop=v_prop, q_id=q_id, i=i, v_exp=v_exp,
                        v_rep=v_rep, tol_abs=p.get('tolerance_abs', 0),
                        tol=p.get('tolerance', 0.01),
                        answer=_("Answer:")
                    ))
            html_content.append('</div>')

        else:
            html_content.append("<div class='options-container'>")
            for i, p in enumerate(props):
                if js_mode:
                    # In JS/QCM mode, we store raw templates + expected-template. 
                    # data-expected will be written by JS after evaluation.
                    v_prop_tpl = to_html(p.get('proposition', ''))
                    v_exp_tpl  = p.get('expected', '')
                    if not '{' in v_exp_tpl:
                        v_exp_tpl = f'{{ {v_exp_tpl} }}'
                    v_rep_tpl  = to_html(p.get('answer', ''))
                    html_content.append(f"""
                <div class='option' id='opt_{q_id}_{i}'
                     data-expected=''
                     data-prop-template='{_escape_attr(v_prop_tpl)}'
                     data-exp-template='{_escape_attr(v_exp_tpl)}'
                     data-rep-template='{_escape_attr(v_rep_tpl)}'
                     onclick='toggleOption(this)'>
                    <input type='checkbox' style='margin-right:12px' onclick='event.stopPropagation()'>
                    <div>
                        <span class='prop-text' id='prop_{q_id}_{i}'></span>
                        <div class='explanation-box' id='expl_{q_id}_{i}'></div>
                    </div>
                </div>""")
                else:
                    v_prop, v_exp, v_rep, v_lab, v_tip = processPropositions(p, q_type, context)
                    v_prop = to_html(v_prop)
                    v_rep = to_html(v_rep)
                    is_exp = "true" if v_exp else "false"
                    html_content.append(f"""
                <div class='option' id='opt_{q_id}_{i}' data-expected='{is_exp}' onclick='toggleOption(this)'>
                    <input type='checkbox' style='margin-right:12px' onclick='event.stopPropagation()'>
                    <div>
                        <span>{v_prop}</span>
                        <div class='explanation-box' id='expl_{q_id}_{i}'>{v_rep}</div>
                    </div>
                </div>""")
            html_content.append("</div>")

        # 4. ACTIONS AND SCORE
        html_content.append("""
            <div id='feedback_{q_id}' style='margin-top:15px; font-weight:bold; color:#2c3e50;'></div>
            <div class='attempts-hint' id='hint_{q_id}'>{attempts} 0 / {max_attempts}</div>
            <div class='btn-group' id='btns_{q_id}'>
                <button class='btn btn-validate' id='val_{q_id}' onclick='validate("{q_id}")'>{submit}</button>
                <button class='btn btn-correct' id='corr_{q_id}' onclick='correct("{q_id}")'>{correction}</button>
                <button class='btn btn-reset' onclick='resetQuestion("{q_id}")'>Reset</button>
            </div>
        </div>""".format(
            q_id=q_id, max_attempts=max_attempts,
            attempts=_("Attempts:"), submit=_("Submit"), correction=_("Correction")
        ))

    # --- JAVASCRIPT ---
# The entire JS block is increased by: 
# - boxMuller(): normal JS generator (Box–Muller) 
# - generateContext(): generates values from JSON specs 
# - resolveSize(): resolves the size (literal or variable name) 
# - evalTemplate(): evaluates {{expr}} in a text with Function() 
# - renderJsCard(): called on the first render AND on Reset to (re)calculate 
# values and update the JS maps DOM. 
# - resetQuestion(): calls renderJsCard() for cards in JS mode. 
# - Initial call: renderJsCard() on all JS cards on loading.

    html_content.append("""
<script>
    let questionScores = {{}};
    const MAX_ATTEMPTS = {max_attempts};

    // -----------------------------------------------------------------------
    // Utilities for "JS template" questions
    // -----------------------------------------------------------------------

    /** Box-Muller: generates a normal value (mean, std). */
    function boxMuller(mean, std) {{
        const u = 1 - Math.random(), v = Math.random();
        return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }}

    /**
    * Resolves size: if sizeExpr is a literal integer → Number, 
    * otherwise search in ctx (variable already drawn).
     */
    function resolveSize(sizeExpr, ctx) {{
        const n = parseInt(sizeExpr, 10);
        if (!isNaN(n)) return n;
        const v = ctx[sizeExpr];
        return Array.isArray(v) ? v.length : (typeof v === 'number' ? Math.round(v) : 1);
    }}

    /**
    * Generates all variables from embedded JSON specs 
    * in map data-variables. 
    * Returns a ctx object {{varName: value_or_array}}.
    */
    function generateContext(varSpecs) {{
        const ctx = {{}};
        for (const [name, spec] of Object.entries(varSpecs)) {{
            const size = resolveSize(spec.size, ctx);
            if (spec.type === 'integers') {{
                if (size === 1) {{
                    ctx[name] = Math.floor(Math.random() * (spec.max - spec.min) + spec.min);
                }} else {{
                    ctx[name] = Array.from({{length: size}},
                        () => Math.floor(Math.random() * (spec.max - spec.min) + spec.min));
                }}
            }} else if (spec.type === 'normal') {{
                if (size === 1) {{
                    ctx[name] = boxMuller(spec.mean, spec.std);
                }} else {{
                    ctx[name] = Array.from({{length: size}}, () => boxMuller(spec.mean, spec.std));
                }}
            }}
        }}
        return ctx;
    }}

                            
    /**
    * Evaluates {{expr}} expressions in template using ctx. 
    * Mirror the Python evaluate_fstring() function: 
    * 
    * 1. In EACH segment (inside AND outside the $...$), protect the 
    * LaTeX commands \cmd{{...}} and Markdown attributes {{...}} 
    * via Unicode sentinels (U+E001/U+E002). 
    * 2. Evaluates via new Function() the remaining {{expr}} (single braces). 
    * 3. Restores sentinels to literal braces.
    *
    * Exemples :
    *   "${{n-1}} x^{{n}}$"  →  "$(n_val-1) x^(n_val)$"  (n calculated)
    *   "\\frac{{1}}{{2}}"   →  "\\frac{{1}}{{2}}"        (preserved)
    *   "{{{{ }}}}"           →  "{{{{ }}}}"               (double → accolades brutes)
    */

    function evalTemplate(template, ctx) {{
        if (!template || !template.includes('{{')) return template;
                        
        const val = template.trim().slice(1, -1).trim().toLowerCase();
        if (val === "true") return "True";
        if (val === "false") return "False";
                        
        function replaceRound(expr) {{
            function process(str) {{
                let result = '', i = 0;
                while (i < str.length) {{
                    const m = str.slice(i).match(/\\bround\s*\(/);
                    if (!m) {{ result += str.slice(i); break; }}

                    const start = i + m.index;
                    result += str.slice(i, start);

                    // Walk through chars counting depth to find the splitting comma and closing paren
                    let depth = 1, comma = -1, j = start + m[0].length;
                    while (j < str.length && depth > 0) {{
                        if      (str[j] === '(')                  depth++;
                        else if (str[j] === ')')                  depth--;
                        else if (str[j] === ',' && depth === 1 && comma === -1) comma = j;
                        if (depth > 0) j++;
                    }}

                    const val = str.slice(start + m[0].length, comma).trim();
                    const digits = str.slice(comma + 1, j).trim();
                    result += `(Math.round((${{process(val)}}) * 10**${{digits}}) / 10**${{digits}})`;
                    i = j + 1;
                }}
                return result;
            }}
            return process(expr);
        }}

        function evalExprPrevious(expr) {{
            expr = expr.trim();
            if (!expr) return undefined;
            // Blocks assignments (k=1) and instructions (;) which do not raise errors in JS
            // if (/[=;]/.test(expr)) return undefined;
            const keys = Object.keys(ctx);
            const vals = keys.map(k => ctx[k]);
            try {{
                // eslint-disable-next-line no-new-func
                return new Function(...keys, 'return (' + expr + ');')(...vals);
            }} catch(e) {{
                return undefined;
            }}
        }}
                        
        function evalExpr(expr) {{
            expr = expr.trim();
            if (!expr) return undefined;
            // if (/[=;]/.test(expr)) return undefined;
                        
            function convertIfElse(expr) {{
                const regex = /(.*)\s+if\s+(.*)\s+else\s+(.*)/;
                const match = expr.match(regex);
                if (!match) return expr;
                const [, val, cond, alt] = match;
                return `(${{cond}}) ? (${{val}}) : (${{alt}})`;
                }}

            // Remplace les fonctions Python courantes par leurs équivalents JS
            const pyToJs = {{
                'abs'  : 'Math.abs',
                'round': 'Math.round',    
                'floor': 'Math.floor',
                'ceil' : 'Math.ceil',
                'min'  : 'Math.min',
                'max'  : 'Math.max',
                'sqrt' : 'Math.sqrt',
                'pow'  : 'Math.pow',
                'log'  : 'Math.log',
                'exp'  : 'Math.exp',
                'sin'  : 'Math.sin',
                'cos'  : 'Math.cos',
                'tan'  : 'Math.tan',
                'int'  : 'Math.trunc',
                'float': 'Number',
                'str'  : 'String',
            }};
            // Cas du round 
            if (expr.includes("round")) expr = replaceRound(expr);

            // cas du log             
            expr = expr.replace(
                /\\blog\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*\)/g,
                (_, x, base) => `(Math.log(${{x}}) / Math.log(${{base}}))`
            );
            // Remplace uniquement les appels de fonctions : abs( → Math.abs(
            // Le \\b évite de remplacer dans des noms de variables comme 'absolute'
            //expr = expr.replace(/\\b(abs|floor|ceil|min|max|sqrt|pow|exp|sin|cos|tan|int|float|str)\\b(?=\s*\()/g,
            //    match => pyToJs[match]
            //);
            expr = expr.replace(
                /(?<!Math\.)\\b(abs|round|int|floor|ceil|min|max|sqrt|pow|log|exp|sin|cos|tan|float|str)\\b(?=\s*\()/g,
                match => pyToJs[match]
            );
                        
            if (expr.includes(" if ")) expr = convertIfElse(expr);
                        
            const keys = Object.keys(ctx);
            const vals = keys.map(k => ctx[k]);
            try {{
                // eslint-disable-next-line no-new-func
                return new Function(...keys, 'return (' + expr + ');')(...vals);
            }} catch(e) {{
                return undefined;
            }}
        }}

        function applyFmt(val, fmt) {{
            // Reproduit les formats Python courants
            if (fmt === null || fmt === undefined) return String(val);
            const num = Number(val);
            if (isNaN(num)) return String(val);  // not numeric → no format

            // {{x:.3f}}  → 3 decimal places fixed
            let m = fmt.match(/^\.(\d+)f$/);
            if (m) return num.toFixed(parseInt(m[1]));

            // {{x:.3e}}  → scientific notation
            m = fmt.match(/^\.(\d+)e$/);
            if (m) return num.toExponential(parseInt(m[1]));

            // {{x:.3g}}  → general notation (removes unnecessary zeros)
            m = fmt.match(/^\.(\d+)g$/);
            if (m) return num.toPrecision(parseInt(m[1])).replace(/\.?0+$/, '');

            // {{x:05d}}  → integer with zeros (padding)
            m = fmt.match(/^0(\d+)d$/);
            if (m) return String(Math.round(num)).padStart(parseInt(m[1]), '0');

            // {{x:5d}}   → integer with spaces (padding)
            m = fmt.match(/^(\d+)d$/);
            if (m) return String(Math.round(num)).padStart(parseInt(m[1]));

            // {{x:+.2f}} → force the sign
            m = fmt.match(/^\+\.(\d+)f$/);
            if (m) return (num >= 0 ? '+' : '') + num.toFixed(parseInt(m[1]));

            return String(val);  // unknown format → fallback
        }}

        function process(text, inMath) {{
            let result = '', i = 0;
            while (i < text.length) {{
                if (text[i] === '{{' && text[i+1] === '{{') {{
                    result += '{{'; i += 2;                        // {{{{ → {{ littéral
                }} else if (text[i] === '}}' && text[i+1] === '}}') {{
                    result += '}}'; i += 2;                        // }}}} → }} littéral
                }} else if (text[i] === '{{') {{
                    // Trouve l'accolade fermante (depth-first)
                    let depth = 1, j = i + 1;
                    while (j < text.length && depth) {{
                        if (text[j] === '{{') depth++;
                        else if (text[j] === '}}') depth--;
                        j++;
                    }}
                    //const expr = text.slice(i+1, j-1);
                    //const val  = evalExpr(expr);
                    const raw  = text.slice(i+1, j-1);
                    const colon = raw.indexOf(":");
                    const expr = colon !== -1 ? raw.slice(0, colon) : raw;
                    const fmt  = colon !== -1 ? raw.slice(colon+1)  : null;
                    const val  = evalExpr(expr);
                    if (val === undefined) {{
                        result += '{{' + expr + '}}';              // unknown → intact
                    }} else if (inMath) {{
                        const last = result.charCodeAt(result.length - 1);
                        const standalone = !result || last === 32 || last === 9 || last === 10;
                        result += standalone ? applyFmt(val, fmt) : '{{' + applyFmt(val, fmt) + '}}';
                    }} else {{
                        result += applyFmt(val, fmt);
                    }}
                    i = j;
                }} else {{
                    result += text[i++];
                }}
            }}
            return result;
        }}

        // Cut on $...$ — treats in AND out
        // template = template.replace(/\\\\(/g, '$').replace(/\\\\)/g, '$'); // becose pandoc
        //template = template.replace(new RegExp('\\\\(', 'g'), '$').replace(new RegExp('\\\\)', 'g'), '$'); // pandoc
        const parts = template.split(/(\$[\s\S]+?\$)/);
        return parts.map((part, idx) =>
            idx % 2 === 0 ? process(part, false)
                            : '$' + process(part.slice(1, -1), true) + '$'
        ).join('');
    }}


    /**
     * Initializes or resets a map in JS-template mode: 
     * - draws new values, 
     * - updates the question text, 
     * - updates each proposal (label, expected, explanation).
     */
    function renderJsCard(id) {{
        const card = document.getElementById('card_' + id);
        if (!card || !card.getAttribute('data-js-template')) return;

        const varSpecs = JSON.parse(card.getAttribute('data-variables') || '{{}}');
        const ctx = generateContext(varSpecs);

        // Question Text
        const qTpl = card.getAttribute('data-question-template') || '';
        const qDiv = document.getElementById('qtext_' + id);
        if (qDiv) qDiv.innerHTML = evalTemplate(qTpl, ctx);

        const qType = card.getAttribute('data-type') || '';

        if (qType.includes('numeric')) {{
            card.querySelectorAll('.numeric-unit').forEach((unit, i) => {{
                const propTpl = unit.getAttribute('data-prop-template') || '';
                const expTpl  = unit.getAttribute('data-exp-template')  || '';
                const repTpl  = unit.getAttribute('data-rep-template')  || '';

                // We first look for a label, or any element whose ID begins with prop_
                const lbl = unit.querySelector('label, [id^="prop_"]');
                if (lbl) lbl.innerHTML = evalTemplate(propTpl, ctx);

                // Calculating Expected Value
                let expVal = evalTemplate(expTpl, ctx);
                try {{ expVal = parseFloat(expVal); }} catch(e) {{}}

                const input = unit.querySelector('.numeric-input');
                if (input) input.setAttribute('data-expected', expVal);

                const expl = unit.querySelector('.explanation-box');
                if (expl) {{
                    const expSpan = expl.querySelector('.exp-val');
                    const repSpan = expl.querySelector('.rep-val');
                    
                    // On remplit les spans s'ils existent (méthode actuelle)
                    if (expSpan) expSpan.innerHTML = expVal;
                    if (repSpan) repSpan.innerHTML = evalTemplate(repTpl, ctx);
                    
                    // OR if one wants to overwrite all the content of the expl with the template (like for options)
                    // expl.innerHTML = evalTemplate(repTpl, ctx);
                }}
            }});
        }} else {{
            card.querySelectorAll('.option').forEach((opt, i) => {{
                const propTpl = opt.getAttribute('data-prop-template') || '';
                const expTpl  = opt.getAttribute('data-exp-template')  || '';
                const repTpl  = opt.getAttribute('data-rep-template')  || '';

                // Text of the proposal
                const propSpan = opt.querySelector('.prop-text');
                if (propSpan) propSpan.innerHTML = evalTemplate(propTpl, ctx);

                // Expected Boolean value
                let expVal = evalTemplate(expTpl, ctx).trim();
                if (expVal === 'True')  expVal = 'true';
                if (expVal === 'False') expVal = 'false';
                opt.setAttribute('data-expected', expVal);

                // We look for the box inside 'opt', regardless of its index i
                const expl = opt.querySelector('.explanation-box'); 
                
                if (expl) {{
                    expl.innerHTML = evalTemplate(repTpl, ctx);
                }} else {{
                    // Secure fallback in case the box is out of the option but with a linked ID 
                    // We get the ID of the option (ex: opt_quiz82_1) and we deduce the ID of the expl
                    const optId = opt.id; // ex: "opt_quiz82_1"
                    const expectedExplId = optId.replace('opt_', 'expl_');
                    const explById = document.getElementById(expectedExplId);
                    if (explById) explById.innerHTML = evalTemplate(repTpl, ctx);
                }}
            }});
        }}

        //if (window.MathJax) MathJax.typesetPromise([card]);
        if (window.MathJax && typeof MathJax.typeset === 'function') MathJax.typeset([card]);
    }}

    // -----------------------------------------------------------------------
    // Global score
    // -----------------------------------------------------------------------
    function updateGlobalScore() {{
        let totalPoints = 0;
        let count = 0;
        for (let key in questionScores) {{
            totalPoints += questionScores[key];
            count++;
        }}
        document.getElementById('total-score').innerText = totalPoints.toFixed(2);
        document.getElementById('total-max').innerText = count;
    }}

    function toggleOption(el) {{
        const card = el.closest('.question-card');
        if (document.getElementById('val_' + card.id.split('_')[1]).disabled) return;
        const cb = el.querySelector('input');
        cb.checked = !cb.checked;
    }}

    function validate(id) {{
        const card = document.getElementById('card_' + id);
        let attempts = parseInt(card.getAttribute('data-attempts')) + 1;
        card.setAttribute('data-attempts', attempts);
        document.getElementById('hint_' + id).innerText = `{attempts} ${{attempts}} / ${{MAX_ATTEMPTS}}`;

        if (card.getAttribute('data-type').includes('numeric')) {{
            let matches = 0;
            const inputs = card.querySelectorAll('.numeric-input');
            inputs.forEach(input => {{
                const val = parseFloat(input.value);
                const exp = parseFloat(input.getAttribute('data-expected'));
                const tolAbs = parseFloat(input.getAttribute('data-tol-abs')) || 0;
                const tolRel = parseFloat(input.getAttribute('data-tol-rel')) || 0.01;
                const tol = Math.max(tolAbs, tolRel * Math.abs(exp));
                if (!isNaN(val) && Math.abs(val - exp) <= tol) matches++;
            }});
            score = matches / inputs.length;
        }} else {{
            let matches = 0;
            const opts = card.querySelectorAll('.option');
            opts.forEach(o => {{
                const checked = o.querySelector('input').checked;
                const expected = o.getAttribute('data-expected') === 'true';
                if (checked === expected) matches++;
            }});
            score = matches / opts.length;
        }}

        questionScores[id] = score;
        updateGlobalScore();
        document.getElementById('feedback_' + id).innerText = "Score : " + score.toFixed(2) + " / 1";

        if (attempts >= MAX_ATTEMPTS) {{
            const btn = document.getElementById('val_' + id);
            btn.disabled = true;
            btn.classList.add('disabled');
        }}
    }}

    function correct(id) {{
        const card = document.getElementById('card_' + id);
        const btnVal = document.getElementById('val_' + id);
        btnVal.disabled = true;
        btnVal.classList.add('disabled');

        card.querySelectorAll('.numeric-input').forEach(i => i.value = i.getAttribute('data-expected'));
        card.querySelectorAll('.explanation-box').forEach(e => e.style.display = 'block');
        card.querySelectorAll('.option').forEach(o => {{
            const cb = o.querySelector('input');
            const exp = o.getAttribute('data-expected') === 'true';
            o.classList.add(cb.checked === exp ? 'match' : 'mismatch');
            cb.checked = exp;
        }});
        if (window.MathJax) MathJax.typeset();
    }}

    function resetQuestion(id) {{
        const card = document.getElementById('card_' + id);

        // 1. Réinitialisation standard
        card.setAttribute('data-attempts', 0);
        document.getElementById('hint_' + id).innerText = `{attempts} 0 / ${{MAX_ATTEMPTS}}`;
        document.getElementById('feedback_' + id).innerText = "";

        const btnVal = document.getElementById('val_' + id);
        btnVal.disabled = false;
        btnVal.classList.remove('disabled');

        // 2. Nettoyage champs numériques
        card.querySelectorAll('.numeric-input').forEach(i => {{
            i.value = "";
            i.style.borderColor = "#dfe6e9";
        }});

        // 3. Nettoyage options QCM
        card.querySelectorAll('.option').forEach(o => {{
            o.querySelector('input').checked = false;
            o.classList.remove('match', 'mismatch');
        }});

        card.querySelectorAll('.explanation-box').forEach(e => e.style.display = 'none');

        // 4. Mélange si non numérique
        /*const type = card.getAttribute('data-type');
        if (type && !type.includes('numeric')) {{
            const container = card.querySelector('.options-container');
            if (container) {{
                let options = Array.from(container.querySelectorAll('.option'));
                for (let i = options.length - 1; i > 0; i--) {{
                    const j = Math.floor(Math.random() * (i + 1));
                    [options[i], options[j]] = [options[j], options[i]];
                }}
                options.forEach(opt => container.appendChild(opt));
            }}
        }}*/
                        
        const type = card.getAttribute('data-type');
        if (type) {{
            const container = card.querySelector('.options-container, .numeric-container');

            if (container) {{
                // On récupère les items (soit .option, soit .numeric-unit)
                let items = Array.from(container.querySelectorAll('.option, .numeric-unit'));
                
                if (items.length > 1) {{
                    // Mélange
                    for (let i = items.length - 1; i > 0; i--) {{
                        const j = Math.floor(Math.random() * (i + 1));
                        [items[i], items[j]] = [items[j], items[i]];
                    }}
                    // Réinsertion
                    items.forEach(item => container.appendChild(item));
                }}
            }}
        }}
                        

        // Removing and re-removing values for JS-template cards
        if (card.getAttribute('data-js-template')) {{
            renderJsCard(id);
        }}
    }}

    // -----------------------------------------------------------------------
    // Initialization on loading — renderJsCard() on all 
    // cards that carry data-js-template.
    // -----------------------------------------------------------------------
    document.addEventListener('DOMContentLoaded', () => {{
        document.querySelectorAll('.question-card[data-js-template]').forEach(card => {{
            const id = card.id.replace('card_', '');
            renderJsCard(id);
        }});
    }});
</script>
</body>
</html>""".format(
        max_attempts=max_attempts,
        attempts=_("Attempts:")
    ))

    return "\n".join(html_content)


# ---------------------------------------------------------------------------
# Small utility function to escape HTML attributes.
# Needed to store raw templates in data-*.
# ---------------------------------------------------------------------------
def _escape_attr(s: str) -> str:
    """Escapes apostrophes and quotation marks for an HTML attribute value."""
    return (s
            #.replace('&', '&amp;')
            .replace("'", '&#39;')
            .replace('"', '&quot;'))

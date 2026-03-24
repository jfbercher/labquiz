import random
import numpy as np
import pandas as pd
from i18n import get_translator
from convert_utils import evaluate_text, safe_eval, processPropositions

# --- CONFIGURATION ---
NB_MAX_ATTEMPTS = 3
rng = np.random.default_rng()

def convert_to_interactive_html(data, lang='en'):
    _ = get_translator(lang)
    html_content = []
    
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
        .explanation-box {{ display: none; margin-top: 10px; padding: 15px; border-radius: 8px; background: #e0fbfa; border-left: 4px solid #00cec9; }}
        .btn-group {{ margin-top: 20px; display: flex; gap: 10px; border-top: 1px solid #eee; padding-top: 15px; }}
        .btn {{ padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; transition: 0.2s; }}
        .btn-validate {{ background: #2ecc71; color: white; }}
        .btn-correct {{ background: #00cec9; color: white; }}
        .btn-reset {{ background: #dfe6e9; color: #636e72; }}
        .disabled {{ opacity: 0.4; pointer-events: none; filter: grayscale(1); }}
        .match {{ background: #eafff0 !important; border-color: #2ecc71 !important; }}
        .mismatch {{ background: #fff0f0 !important; border-color: #e74c3c !important; }}
        .attempts-hint {{ font-size: 0.85em; color: #7f8c8d; margin-top: 5px; }}
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
        if q_id == "title" or not isinstance(q_content, dict): continue
        
        # 1. CONTEXT GENERATION (FOR TEMPLATES)
        context = {}
        q_type = str(q_content.get('type', 'qcm')).lower()
        if "template" in q_type:
            variables = q_content.get('variables', {})
            for var_name in variables.keys():
                engine = variables[var_name].get('engine')
                engine_call = variables[var_name].get('call')
                engine_prefix = "rng." if engine == "numpy rng." else "pd."
                expression = f"{engine_prefix}{engine_call}"
                context[var_name] = safe_eval(expression)

        # 2. START CARD
        html_content.append(f"<div class='question-card' id='card_{q_id}' data-type='{q_type}' data-attempts='0'>")
        
        q_text = evaluate_text(q_content.get('question', ''), context)
        html_content.append(f"<div class='question-text'>{q_text}</div>")
        
        props = q_content.get('propositions', [])
        random.shuffle(props)


        # 3. QUESTION CONTENT
        if "numeric" in q_type:
            for i, p in enumerate(props):
                v_prop, v_exp, v_rep, v_lab = processPropositions(p, q_type, context)
                html_content.append("""
                <div class='numeric-unit'>
                    <label>{v_prop}</label><br>
                    <input type='number' step='any' class='numeric-input' id='input_{q_id}_{i}' 
                           data-expected='{v_exp}' data-tol-abs='{tol_abs}' data-tol-rel='{tol}'>
                    <div class='explanation-box' id='expl_{q_id}_{i}'><b>{answer}</b> {v_exp}<br>{v_rep}</div>
                </div>""".format(v_prop=v_prop, q_id=q_id, i=i, v_exp=v_exp, 
                                 v_rep=v_rep, tol_abs=p.get('tolerance_abs', 0), 
                                 tol=p.get('tolerance', 0.01),
                                 answer = _("Answer:")
                                 )
                )

        else:
            html_content.append("<div class='options-container'>")
            for i, p in enumerate(props):
                v_prop, v_exp, v_rep, v_lab = processPropositions(p, q_type, context)
                is_exp = "true" if v_exp else "false"
                #print("debug:", v_exp, type(v_exp), is_exp)
                #print("--> checkboxes", p.get('expected', ''), v_exp, is_exp)

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
            <div class='attempts-hint' id='hint_{q_id}'>{attempts} 0 / {NB_MAX_ATTEMPTS}</div>
            <div class='btn-group' id='btns_{q_id}'>
                <button class='btn btn-validate' id='val_{q_id}' onclick='validate("{q_id}")'>{submit}</button>
                <button class='btn btn-correct' id='corr_{q_id}' onclick='correct("{q_id}")'>{correction}</button>
                <button class='btn btn-reset' onclick='resetQuestion("{q_id}")'>Reset</button>
            </div>
        </div>""".format(q_id=q_id, NB_MAX_ATTEMPTS=NB_MAX_ATTEMPTS, 
                         attempts=_("Attempts:"), submit=_("Submit"), correction=_("Correction")
                         )
        )

    # --- JAVASCRIPT ---
    html_content.append("""
<script>
    let questionScores = {{}};
    const MAX_ATTEMPTS = {NB_MAX_ATTEMPTS};

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
            // On cible directement les INPUTS qui possèdent les data-expected
            const inputs = card.querySelectorAll('.numeric-input'); 
            inputs.forEach(input => {{
                const val = parseFloat(input.value);
                const exp = parseFloat(input.getAttribute('data-expected'));
                const tolAbs = parseFloat(input.getAttribute('data-tol-abs')) || 0;
                const tolRel = parseFloat(input.getAttribute('data-tol-rel')) || 0.01;
                const tol = Math.max(tolAbs, tolRel * Math.abs(exp));

                // On ne compte le point que si la valeur n'est PAS NaN ET qu'elle est dans la tolérance
                if (!isNaN(val) && Math.abs(val - exp) <= tol) {{
                    matches++;
                    //input.style.borderColor = '#2ecc71'; // Vert si juste
                }} 
            }});
            score = matches / inputs.length;
        }}
        else {{
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
        // Désactivation validation
        const btnVal = document.getElementById('val_' + id);
        btnVal.disabled = true;
        btnVal.classList.add('disabled');

        // Affichage correction
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
        
        // 1. Réinitialisation standard (Essais, bouton, feedback)
        card.setAttribute('data-attempts', 0);
        document.getElementById('hint_' + id).innerText = `{attempts} 0 / ${{MAX_ATTEMPTS}}`;
        document.getElementById('feedback_' + id).innerText = "";
        
        const btnVal = document.getElementById('val_' + id);
        btnVal.disabled = false;
        btnVal.classList.remove('disabled');

        // 2. Nettoyage des champs numériques
        card.querySelectorAll('.numeric-input').forEach(i => {{
            i.value = ""; 
            i.style.borderColor = "#dfe6e9";
        }});

        // 3. Nettoyage des options (QCM/QCU)
        card.querySelectorAll('.option').forEach(o => {{
            o.querySelector('input').checked = false;
            o.classList.remove('match', 'mismatch');
        }});

        card.querySelectorAll('.explanation-box').forEach(e => e.style.display = 'none');

        // 4. MÉLANGE SI NON NUMÉRIQUE
        // On vérifie si data-type ne contient PAS "numeric"
        const type = card.getAttribute('data-type');
        if (type && !type.includes('numeric')) {{
            // On cherche le conteneur des options (la div qui contient les .option)
            const container = card.querySelector('.options-container') ;
            if (container) {{
                let options = Array.from(container.querySelectorAll('.option'));
                // Mélange de Fisher-Yates
                for (let i = options.length - 1; i > 0; i--) {{
                    const j = Math.floor(Math.random() * (i + 1));
                    // On échange dans le tableau
                    [options[i], options[j]] = [options[j], options[i]];
                }}
                // On les ré-attache au conteneur (appendChild déplace l'élément s'il existe déjà)
                options.forEach(opt => container.appendChild(opt));
            }}
        }}
    }}

    function resetQuestionOld(id) {{
        const card = document.getElementById('card_' + id);
        card.setAttribute('data-attempts', 0);
        document.getElementById('hint_' + id).innerText = `{attempts} 0 / ${{MAX_ATTEMPTS}}`;
        document.getElementById('feedback_' + id).innerText = "";
        
        const btnVal = document.getElementById('val_' + id);
        btnVal.disabled = false;
        btnVal.classList.remove('disabled');

        card.querySelectorAll('.numeric-input').forEach(i => {{ i.value = ""; i.style.borderColor = "#dfe6e9"; }});
        card.querySelectorAll('.explanation-box').forEach(e => e.style.display = 'none');
        card.querySelectorAll('.option').forEach(o => {{
            o.querySelector('input').checked = false;
            o.classList.remove('match', 'mismatch');
        }});
    }}
</script>
</body>
</html>""".format(
    NB_MAX_ATTEMPTS=NB_MAX_ATTEMPTS, attempts=_("Attempts:")
))
    return "\n".join(html_content)
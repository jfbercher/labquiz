#import re
#import numpy as np
#import pandas as pd

#from i18n import get_translator
from convert_utils import (evaluate_fstring, fix_latex_syntax, 
                           _template_to_py, process_braces, 
                           processPropositions, looks_like_markdown, micro_text_cleaning)
import pypandoc
import re


# ---------------------------------------------------------------------------
# LaTeX conversion
# ---------------------------------------------------------------------------

def to_LaTeX(markdown_text):
    """Converts markdown text to LaTeX via pypandoc. Returns plain text if not markdown."""
    if looks_like_markdown(markdown_text):
        markdown_text = fix_latex_syntax(markdown_text)
        latex = pypandoc.convert_text(
            markdown_text,
            to="latex",
            format="md",
            extra_args=["--wrap=none", '--extension=link_attributes']
        )
        latex = latex.replace("\n\n", "\n").rstrip()
        if 'longtable' in latex:  # correct longtable parameters
            latex = re.sub( r'\\begin\{longtable\}\[\]\{@\{\}(.*?)@\{\}\}',
                lambda m: r'\begin{longtable}{|' + '|'.join(list(m.group(1))) + '|}', latex)
        if '{figure}' in latex: # for our questions, convert figure envs to center ones - no numembering, no floating
            latex = re.sub(
                r'\\begin{figure}\s*\\centering\s*(.*?)\s*\\caption{(.*?)}\s*\\end{figure}',
                r'\\begin{center}\n\1\n\\\\[-0.6em]\n{\\small Figure:~ \\itshape \2}\n\\end{center}',
                latex, flags=re.S)
        return latex
    else:
        return micro_text_cleaning(markdown_text)

# ---------------------------------------------------------------------------
# Normalise the 'expected' YAML field to a bare Python expression.
# ---------------------------------------------------------------------------

def _extract_expected_expr(raw):
    """
    Normalises the 'expected' field to a bare Python expression string.

    Examples:
        "n"           → "n"
        n*(n-1)       → "n*(n-1)"
        "{n*(n-1)}"   → "n*(n-1)"
        "'{n*(n-1)}'" → "n*(n-1)"
    """
    s = str(raw).strip().strip("'").strip('"').strip()
    # Strip surrounding { } if present (f-string style)
    if s.startswith('{') and s.endswith('}'):
        s = s[1:-1].strip()
    return s


# ---------------------------------------------------------------------------
# Generate \pyc{} lines to embed inside the question block.
# PythonTeX re-executes \pyc{} for each AMC copy → different values per copy.
# ---------------------------------------------------------------------------

def _make_pyc_lines(variables, propositions, q_type):
    """
    Generates \\pyc{} lines to embed inside the \\element{} block.

    Contains:
      1. Variable draws from YAML definitions.
      2. For numeric questions: expected_i, _nbch_i, _ndec_i, _ndigits_i.
         _ndigits = _nbch + _ndec is the total digit count required by AMC.
      3. For non-numeric template questions: v_exp_i (bool) per proposition,
         used by \\ifthenelse{\\boolean{!{v_exp_i}}} to select the correct tag.

    Args:
        variables:    dict of {var_name: {engine, call}} from YAML.
        propositions: list of proposition dicts from YAML.
        q_type:       question type string (e.g. 'numeric-template').

    Returns:
        List of LaTeX strings, each of the form \\pyc{...}.
    """
    lines = []

    # 1. Variable draws
    for var_name, var_def in variables.items():
        prefix = 'rng.' if var_def.get('engine') == 'numpy rng.' else 'pd.'
        lines.append(f'  \\pyc{{{var_name} = {prefix}{var_def.get("call", "")}}}')

    # 2. Numeric: expected values and digit metadata
    if 'numeric' in q_type:
        for i, p in enumerate(propositions):
            expr   = _extract_expected_expr(p.get('expected', '0'))
            expr = expr.replace('% 2 == 0', '& 1 == 0').replace('% 2 != 0', '& 1 != 0') #small issue correction - there might be others!
            suffix = f'_{i}' if len(propositions) > 1 else ''
            lines.append(f'  \\pyc{{expected{suffix} = {expr}}}')
            lines.append(f'  \\pyc{{_nbch{suffix} = len(str(abs(int(expected{suffix}))))}}')
            lines.append(
                f'  \\pyc{{_ndec{suffix} = len(str(expected{suffix}).split(".")[-1].rstrip("0"))'
                f' if "." in str(expected{suffix}) else 0}}'
            )
            # Total digits = integer digits + decimal digits (required by AMC)
            lines.append(f'  \\pyc{{_ndigits{suffix} = _nbch{suffix} + _ndec{suffix}}}')

    # 3. Non-numeric template: boolean expected per proposition
    else:
        for i, p in enumerate(propositions):
            expr = _extract_expected_expr(p.get('expected', 'False'))
            expr = expr.replace('% 2 == 0', '& 1 == 0').replace('% 2 != 0', '& 1 != 0') #small issue correction - there might be others!
            lines.append(f'  \\pyc{{v_exp_{i} = bool({expr})}}')

    return lines


# ---------------------------------------------------------------------------
# Global \begin{pycode} block: imports + rng, placed once in the document.
# ---------------------------------------------------------------------------

def make_global_pycode_header():
    """
    Returns the global \\begin{pycode} block to place once in the document
    (before \\onecopy (english) or \\exemplaire (french)). Contains only imports and rng initialisation.
    The rng instance is shared across all \\pyc{} calls in the document,
    ensuring independent draws for each question instantiation by AMC.
    """
    return (
        '\\begin{pycode}\n'
        'import numpy as np\n'
        'import pandas as pd\n'
        'rng = np.random.default_rng()\n'
        '\\end{pycode}'
    )


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def convert_to_amc_pytex(data, use_negative_points=True, output_scoring=False):
    """
    Converts a quiz YAML dict to AMC-LaTeX with PythonTeX support.

    For template questions:
      - Variable draws and expected value computations are placed as \\pyc{}
        lines INSIDE the \\element{} block → AMC re-executes them per copy.
      - Question text and proposition text: {expr} → \\py{expr}.
      - Non-numeric propositions: \\ifthenelse{\\boolean{!{v_exp_i}}} selects
        \\correctchoice or \\wrongchoice dynamically based on the drawn values.
      - Numeric propositions: wrapped in \\pys{\\AMCnumericChoices{!{expected}}
        {digits=!{_ndigits},decimals=!{_ndec}}}.

    For non-template questions: identical to the static exporter.

    The caller must place make_global_pycode_header() once in the document
    (before \\onecopy (english) or \\exemplaire (french)).

    Args:
        data:                quiz dict (as loaded from YAML).
        use_negative_points: if True, wrong choices get a negative score.
        output_scoring:      if True, append \\scoring{b=,m=} to each choice.

    Returns:
        A LaTeX string ready to be included in an AMC+PythonTeX document.
    """
    latex_output = []

    for q_id, q_content in data.items():
        if q_id == 'title' or not isinstance(q_content, dict): continue

        q_type      = str(q_content.get('type', '')).lower()
        is_template = 'template' in q_type
        variables   = q_content.get('variables', {})
        props       = q_content.get('propositions', [])
        var_names   = list(variables.keys())
        context     = {}  # empty for non-template; template uses \py{} instead

        if is_template:
            latex_output.append(f'\n% --- {q_id} (pythontex template) ---')
            q_text = to_LaTeX(_template_to_py(q_content.get('question', ''), var_names))

        else:
            latex_output.append(f'\n% --- {q_id} ---')
            q_text = to_LaTeX(evaluate_fstring(q_content.get('question', ''), context))

        q_category = q_content.get('category', 'nocategory')
        q_label    = q_content.get('label', f'q:{q_id}')

        # Numeric questions must use questionmult (AMC requirement).
        # For templates and calculated answers, we cannot know the number of correct answers (that can be zero)
        # thus 'questionmult' is required (allows zero correct answer).
        if 'numeric' in q_type:
            amc_tag = 'questionmultx'
        else:
            is_mult = ('multiple' in q_type or
                       len([p for p in props if p.get('expected') is True]) > 1)
            amc_tag = 'questionmult' if is_mult or is_template else 'question'

        # --- Question block ---
        latex_output.append(f'\\element{{{q_category}}}{{')
        latex_output.append(f'  \\begin{{{amc_tag}}}{{{q_label}}}')

        # \pyc{} lines go first, before the question text, so variables are
        # available when \py{} expressions in the text are evaluated.
        if is_template:
            latex_output.extend(_make_pyc_lines(variables, props, q_type))

        #latex_output.append(f'  {q_text}') #First version
        #This enables to actually expand the fstring even with format marks
        q_text = process_braces(q_text)
        latex_output.append(f'  \\py{{%')
        latex_output.append(f'  rf"{q_text}"')
        latex_output.append(f'     }}')


        # --- Propositions ---
        if 'numeric' in q_type:
            # One \AMCnumericChoices per proposition.
            # digits = total digit count (integer + decimal), as required by AMC.
            '''
             {%
  \def\AMCbeginQuestion#1#2{}%
  \AMCquestionNumberfalse
      \begin{questionmult}{partie1}
        Quel est le premier résultat ?
        \AMCnumericChoices{123}{digits=3}
      \end{questionmult}
  }
            '''
            is_multiple = len(props) > 1
            if is_multiple:
                for i, p in enumerate(props):
                    suffix = f'_{i}' if len(props) > 1 else ''
                    if is_template:
                        v_prop = to_LaTeX(_template_to_py(p.get('proposition', ''), var_names))
                        v_prop = process_braces(v_prop)
                        latex_output.append('  {%')
                        latex_output.append('  \def\AMCbeginQuestion#1#2{}%')
                        latex_output.append('  \AMCquestionNumberfalse')
                        latex_output.append(f'    \\begin{{questionmultx}}{{{q_label}{suffix}}}')
                        latex_output.append(v_prop)
                        latex_output.append(
                            f'      \\pys{{\\AMCnumericChoices{{!{{expected{suffix}}}}}'
                            f'{{digits=!{{_ndigits{suffix}}},decimals=!{{_ndec{suffix}}}}}}}'
                        )
                        latex_output.append('  \end{questionmultx}')
                        latex_output.append('  }')
                    else:
                        # Static: evaluate as before
                        v_prop, v_exp, v_ans, _, v_tip = processPropositions(p, q_type, context)
                        latex_output.append('  {%')
                        latex_output.append('  \def\AMCbeginQuestion#1#2{}%')
                        latex_output.append('  \AMCquestionNumberfalse')
                        latex_output.append(f'    \\begin{{questionmultx}}{{{q_label}{suffix}}}')
                        latex_output.append(v_prop)
                        nbch    = len(str(abs(int(val_eval))))
                        ndec    = len(str(val_eval).split('.')[-1].rstrip('0')) if '.' in str(val_eval) else 0
                        ndigits = nbch + ndec
                        latex_output.append(
                            f'  \\AMCnumericChoices{{{val_eval}}}'
                            f'{{digits={ndigits},decimals={ndec}}}'
                        )
                        latex_output.append('  \end{questionmultx}')
                        latex_output.append('  }')
            else:  # single proposition in numeric
                suffix = ''
                if is_template:
                    v_prop = to_LaTeX(_template_to_py(p.get('proposition', ''), var_names))
                    v_prop = process_braces(v_prop)
                    latex_output.append(v_prop)
                    latex_output.append(
                        f'    \\pys{{\\AMCnumericChoices{{!{{expected{suffix}}}}}'
                        f'{{digits=!{{_ndigits{suffix}}},decimals=!{{_ndec{suffix}}}}}}}'
                    )
                else:
                    # Static: evaluate as before
                    v_prop, v_exp, v_ans, _, v_tip = processPropositions(p, q_type, context)
                    latex_output.append(v_prop)
                    nbch    = len(str(abs(int(val_eval))))
                    ndec    = len(str(val_eval).split('.')[-1].rstrip('0')) if '.' in str(val_eval) else 0
                    ndigits = nbch + ndec
                    latex_output.append(
                        f'  \\AMCnumericChoices{{{val_eval}}}'
                        f'{{digits={ndigits},decimals={ndec}}}'
                    )
    
        else:
            latex_output.append('    \\begin{choices}')
            for i, p in enumerate(props):
                if is_template:
                    # Proposition text and metadata: {expr} → \py{expr}
                    v_prop = to_LaTeX(_template_to_py(p.get('proposition', ''), var_names))
                    v_prop = process_braces(v_prop)

                    v_ans  = process_braces(to_LaTeX(_template_to_py(p.get('answer', ''), var_names)))
                    v_tip  = process_braces(to_LaTeX(_template_to_py(p.get('tip',    ''), var_names)))

                    latex_output.append(f'\\py{{%')
                    latex_output.append(f'    rf"\\correctchoice{{{{ {v_prop} }}}}"')
                    latex_output.append(f'    if v_exp_{i}')
                    latex_output.append(f'    else')
                    latex_output.append(f'    rf"\\wrongchoice{{{{ {v_prop} }}}}"')
                    latex_output.append(f'    }}')

                else:
                    v_prop, v_exp, v_ans, _, v_tip = processPropositions(p, q_type, context)
                    v_prop     = to_LaTeX(v_prop)
                    is_correct = v_exp
                    cmd        = '\\correctchoice' if is_correct else '\\wrongchoice'
                    bonus      = p.get('bonus', 1 if is_correct else 0)
                    malus      = p.get('malus', -1 if (use_negative_points and not is_correct) else 0)
                    malus      = -abs(malus)
                    if output_scoring:
                        latex_output.append(f'      {cmd}{{{v_prop}}} \\scoring{{b={bonus},m={malus}}}')
                    else:
                        latex_output.append(f'      {cmd}{{{v_prop}}}')

                latex_output.append(f'      %answer: {v_ans}')
                latex_output.append(f'      %tip: {v_tip}')

            latex_output.append('    \\end{choices}')

        latex_output.append(f'  \\end{{{amc_tag}}}')
        latex_output.append('}')

    return '\n'.join(latex_output)

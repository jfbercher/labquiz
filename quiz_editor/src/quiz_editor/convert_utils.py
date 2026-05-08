import re
import numpy as np
import pandas as pd
import html
import math
import html
from pathlib import Path
import base64
import markdown
#import pypandoc


rng = np.random.default_rng()

# Restricted eval namespace: restricted builtins, only numpy and math
safe_globals = {"__builtins__": {
    "int": int,
    "float": float,
    "bool": bool,
    "len": len,
    "str": str,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow
    }, "np": np, "math": math}


def looks_like_markdown(text: str) -> bool:
    #pattern = r"(\*{1,2}|_{1,2}|`)" # simple
    #improved by requiring symbols to be balanced
    pattern = r"(\*\*[^*]+\*\*|\*[^*]+\*|__[^_]+__|_[^_]+_|`[^`]+`)" # detects bold, italic, and inline code
    pattern_links = r'!\[.*?\]\(.*?\)|\[[^\]]+\]\([^)]+\)' # detects links

    return bool(re.search(pattern, text)) or bool(re.search(pattern_links, text))

def micro_text_cleaning(text):
    # Clean up special characters without requiring full conversion
    if not text.lstrip().startswith("%"):
        text = text.replace("%", r"\%")
    text = text.replace("&", r"\&")
    return text

def fix_latex_syntax(text):
    """
    Standardizes LaTeX delimiters for Streamlit:
    \( math \) -> $math$
    \[ math \] -> $$math$$
    """
    # 1. Replace block delimiters \[ ... \] with $$ ... $$
    # Flags=re.DOTALL allows the regex to match across multiple lines
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    
    # 2. Replace inline delimiters \( ... \) with $ ... $
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text)
    
    return text


def normalize_md(text):
    # Lists: Pattern to detect any start of item: - or * or 1.
    list_item = r'\s*(?:[-*]|\d+\.)\s+'
    # Images
    image_item = r'!\[.*?\]\(.*?\)(?:\{.*?\})?'

    # 1. Before the block: Text followed by the start of a list 
    # # We check that the current line is not already a list or empty
    pattern_before = rf'(^|(?<=\n))(?!{list_item}|\s*$)(.+)\n(?={list_item})'
    text = re.sub(pattern_before, r'\1\2\n\n', text)
    
    # 2. After the block: End of list followed by text 
    # # We check that the following line is not a list or empty
    pattern_after = rf'(\n{list_item}.*$)\n(?!{list_item}|\s*$)'
    text = re.sub(pattern_after, r'\1\n\n', text, flags=re.MULTILINE)


    # Before: Text followed by an image 
    # # We look for a character, a line break, then the image
    text = re.sub(rf'([^\n])\n(?={image_item})', r'\1\n\n', text)

    # After: Image followed by text 
    # # We look for the image, a line break, then text that is not an empty line
    text = re.sub(rf'({image_item}.*)\n(?!\s*$)', r'\1\n\n', text)
    
    text = text.replace('\n\n\n', '\n\n')
    
    return text


def has_markdown_img_link(text: str) -> bool:
    pattern_links = r'!\[.*?\]\(.*?\)|\[[^\]]+\]\([^)]+\)' # detects links
    return  bool(re.search(pattern_links, text))

def convert_markdown_images_to_base64(markdown_text, base_path="."):
    """
    Scans markdown for local image syntax ![alt](path) and replaces 
    the path with a Base64 data URI.
    """
    # Regex pattern for ![alt](path)
    # Group 1: alt text, Group 2: path/url
    pattern = r'\!\[(.*?)\]\((.*?)\)'
    
    def replacer(match):
        alt_text = match.group(1)
        path_str = match.group(2)
        
        # 1. If it's a web URL, return the original match (do nothing)
        if path_str.startswith(('http://', 'https://', 'data:')):
            return match.group(0)
        # 2. Check if the local file exists
        img_path = Path(base_path) / path_str
        if img_path.is_file():
            # Perform conversion to Base64
            with open(img_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
            extension = img_path.suffix.lower().replace('.', '')
            # Default to png if extension is missing/weird
            mime_type = extension if extension in ['png', 'jpg', 'jpeg', 'gif', 'webp'] else 'png'
            # Return the new markdown string with embedded data
            return f'![{alt_text}](data:image/{mime_type};base64,{encoded_string})'
        # 3. If file not found, return original string (nothing to replace)
        return match.group(0)
    # If the pattern is found, replacer() is called; otherwise, text remains unchanged
    return re.sub(pattern, replacer, markdown_text)


def convert_markdown_images_to_html(markdown_text, base_path="."):
    """
    Converts ![alt](path){width=nn% height=mm%} into an HTML <img> tag.
    dimensions using both HTML attributes and CSS object-fit
    AND embed the image by encoding it in base64.
    """
    pattern = r'\!\[(.*?)\]\((.*?)\)(?:\{(.*?)\})?'
    
    def replacer(match):
        alt_text = match.group(1)
        path_str = match.group(2)
        attributes = match.group(3)
        
        # 1. If it's a web URL, return the original match (do nothing)
        if path_str.startswith(('http://', 'https://', 'data:')):
            return match.group(0)
        
        # 2. Check if the local file exists
        img_path = Path(base_path) / path_str
        
        if img_path.is_file():
            # Perform conversion to Base64
            with open(img_path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            
            # Default to png if extension is missing/weird
            ext = img_path.suffix.lower().replace('.', '')
            mime = ext if ext in ['png', 'jpg', 'jpeg', 'gif', 'webp'] else 'png'
            
            # Default style: ensure it doesn't break the container
            # object-fit: contain ensures the image isn't stretched
            style_parts = ["max-width: 100%;", "object-fit: contain;"]
            html_attrs = []

            if attributes:
                w_match = re.search(r'width=(\d+%|\d+px|\d+)', attributes)
                h_match = re.search(r'height=(\d+%|\d+px|\d+)', attributes)
                
                if w_match:
                    w = w_match.group(1)
                    val = w + ('px' if w.isdigit() else '')
                    style_parts.append(f"width: {val} !important;")
                    html_attrs.append(f'width="{val}"')
                if h_match:
                    h = h_match.group(1)
                    val = h + ('px' if h.isdigit() else '')
                    style_parts.append(f"height: {val} !important;")
                    html_attrs.append(f'height="{val}"')

            style_str = f' style="{" ".join(style_parts)}"'
            attr_str = f' {" ".join(html_attrs)}' if html_attrs else ""

             # Return the <img> string with embedded data and style
            return f'<img src="data:image/{mime};base64,{data}" alt="{alt_text}"{attr_str}{style_str}>'
        # 3. If file not found, return original string (nothing to replace)
        return match.group(0)
    # If the pattern is found, replacer() is called; otherwise, text remains unchanged
    return re.sub(pattern, replacer, markdown_text)


def to_html(text, embedImages=True, protectMaths=True):
    # pypandoc fait vraiment des misères avec -f markdown+tex_math_dollars
    # ou le --mathjax - D'où l'extraction puis réinsertion des maths
    if not looks_like_markdown(text):
        return text

    if embedImages and has_markdown_img_link(text):
        text = convert_markdown_images_to_base64(text)

    # Extra markdown normalization    
    text = normalize_md(text)

    # Extrait et protège les blocs maths
    placeholders = {}
    counter = 0

    def protect(m):
        nonlocal counter
        key = f"MATHPLACEHOLDER{counter}X"
        placeholders[key] = m.group(0)
        counter += 1
        return key

    patterns = [
        r'\$\$[\s\S]+?\$\$',          # display $$...$$
        r'\\\[[\s\S]+?\\\]',           # display \[...\]
        r'\$[^\$]+?\$',                # inline $...$
        r'\\\(.+?\\\)',                # inline \(...\)
    ]
    protected = text
    if protectMaths:
        for pattern in patterns:
            protected = re.sub(pattern, protect, protected, flags=re.DOTALL)

    zz = '''html = pypandoc.convert_text(
        protected,
        to="html",
        format='markdown+pipe_tables+implicit_figures',  # or 'gfm',
        extra_args=["--wrap=none"] 
    )''' 

    html = markdown.markdown(protected, extensions=['tables', 'fenced_code', 'attr_list', 'nl2br'])

    html = html.replace("\n\n", "\n").rstrip()

    if protectMaths:
        # Restore maths
        for key, val in placeholders.items():
            html = html.replace(key, val)

    # Add style for tables
    html = html.replace('<table>', '<table class="styled">')

    return html

def to_html_old(text):
    return markdown.markdown(text, extensions=['tables', 'fenced_code', 'nl2br'])


def strip_f_prefix(template: str) -> str:
    """Removes the 'f' prefix from the template string."""
    import re
    return re.sub(r'^\s*f([\'"]{1,3})', r'\1', template, count=1)


def evaluate_fstring_previous_again(template, context):
    # New version also take into account true LaTeX in propositions/questions
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

    safe_globals = {
        "__builtins__": {},
        "np": np,
    }
    
    val = eval("f" + repr(template), safe_globals, context).strip("'").strip('"')
    return val


def evaluate_fstring_avant(template, context):
    """
    Evaluates a template string containing {expr} placeholders, with special
    handling for LaTeX math segments delimited by $...$ (inline) or $$...$$ (display).

    Differences from a standard Python f-string:
      - Unknown expressions (not in context) are left intact rather than raising KeyError.
      - Inside math delimiters, curly braces are preserved when needed to keep LaTeX valid
        (e.g. x^{n} → x^{12}, not x^12).
      - {{ and }} are treated as literal { and } in both math and non-math segments.
      - Both $...$ and $$...$$ are supported; $$...$$ is matched first to avoid
        being incorrectly split by the $...$ pattern.

    Args:
        template: the template string, optionally prefixed with f' or f\".
        context:  dict of variable names → values available for substitution.

    Returns:
        The template with all resolvable {expr} replaced by their values.
    """
    if not isinstance(template, str): return template
    template = strip_f_prefix(template)
    if '{' not in template: return template

    # Restricted eval namespace: restricted builtins, only numpy and math
    safe_globals = {"__builtins__": {
        "int": int,
        "float": float,
        "bool": bool,
        "len": len,
        "str": str,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        }, "np": np, "math": math}

    def eval_expr(expr):
        """Evaluates expr in safe_globals + context. Returns None on any error."""
        try: return eval(expr, safe_globals, context)
        except Exception: return None

    def _get_fmt(text):
        res = text.split(':',1)
        if len(res) > 1:
            expr, fmt = res[0], res[1]
        else:
            expr, fmt = text, None
        return expr, fmt

    def _apply_fmt(val, fmt):
        """Applies an optional format spec to a value."""
        if val is None: return None
        try:
            return format(val, fmt) if fmt is not None else str(val)
        except Exception:
            return str(val)

    def process(text, in_math):
        """
        Substitutes {expr} placeholders in a text segment.

        Two behaviours depending on in_math:
          - in_math=False (outside math delimiters):
              {expr}  → str(value)      if evaluable
              {expr}  → {expr}          if unknown (left intact)
              {{      → {               literal brace
              }}      → }               literal brace

          - in_math=True (inside $...$ or $$...$$):
              {expr}  → {value}         if preceded by a LaTeX operator (^, _, \cmd...)
                                        so the LaTeX structure stays valid
              {expr}  → value           if standalone (start of segment or preceded
                                        by whitespace), e.g. ${n} x^{n-1}$
              {expr}  → {expr}          if unknown (left intact, e.g. \sum_{k=1})
              {{      → {               literal brace (e.g. \sum_{{k=1}} → \sum_{k=1})
              }}      → }               literal brace
        """
        result, i = [], 0
        while i < len(text):
            c = text[i]
            if c == '{' and i+1 < len(text) and text[i+1] == '{':
                # {{ → literal {
                result.append('{'); i += 2
            elif c == '}' and i+1 < len(text) and text[i+1] == '}':
                # }} → literal }
                result.append('}'); i += 2
            elif c == '{':
                # Find the matching closing brace (handles nesting)
                depth, j = 1, i+1
                while j < len(text) and depth:
                    depth += (text[j] == '{') - (text[j] == '}'); j += 1
                if depth != 0:
                    # No matching closing brace found (e.g. { split across math segments)
                    # Leave the { intact and move on
                    result.append('{'); i += 1
                    continue
                raw = text[i+1:j-1]
                expr, fmt = _get_fmt(raw)
                #expr, val = text[i+1:j-1], 
                val = eval_expr(expr)
                if val is None:
                    # Unknown expression: leave intact
                    result.append('{' + expr + '}')
                elif not in_math:
                    # Outside math: substitute bare value
                    result.append(_apply_fmt(val, fmt))
                else:
                    # Inside math: preserve braces unless standalone
                    last = ''.join(result)
                    standalone = not last or last[-1] in ' \t\n'
                    result.append(str(val) if standalone else '{' + _apply_fmt(val, fmt) + '}')
                i = j
            else:
                result.append(c); i += 1
        return ''.join(result)

    # Split on $$...$$ (display math) and $...$ (inline math), in that order.
    # $$...$$ must be tried first to avoid being incorrectly consumed by $...$.
    # The capturing group keeps the delimiters in the parts list.
    # Even indices → outside math, odd indices → math segment (delimiters included).
    parts = re.split(r'(?<!\\)(\$\$.+?\$\$|\$[^\$]+?\$)', template, flags=re.DOTALL)
    out = ''.join(
        process(p, False) if i % 2 == 0
        else process(p[2:-2], True).join(('$$', '$$')) if p.startswith('$$')
        else process(p[1:-1], True).join(('$', '$'))
        for i, p in enumerate(parts)
    )
    #print("out", out)
    return out


# ---------------------------------------------------------------------------
# Template substitution: {expr} → \py{expr} for known/evaluable expressions,
# intact otherwise. This identify expressions, tests if they are evaluable
# and then "protects" them by a \py{expr}; this ensures that it survives 
# conversion to html and LaTeX via pandoc. A small pre or post-processing (process-braces)
# can be applied to get evaluate_fstring (post), or conversion to LaTeX (post) / html (pre)
# ---------------------------------------------------------------------------

def _template_to_py(template, var_names):
    """
    Converts a template string to a PythonTeX-ready LaTeX string.

    Substitution rules (same segmentation as evaluate_fstring):
      - Outside $...$:
          {expr}  → \py{expr}    if expr involves only known variables/builtins
          {expr}  → {expr}       otherwise (left intact)
          {{      → {            literal brace
          }}      → }            literal brace

      - Inside $...$:
          {expr}  → {\py{expr}}  if preceded by a LaTeX operator (^, _, etc.)
          {expr}  → \py{expr}    if standalone (start or preceded by whitespace)
          {expr}  → {expr}       if unknown (left intact, e.g. \sum_{k=1})
          {{      → {            literal brace
          }}      → }            literal brace

    Args:
        template:  the raw template string (may have f' prefix).
        var_names: list of variable names defined via \\pyc{} in the question.
    """
    if not isinstance(template, str): return template
    # Strip f-prefix if present
    template = re.sub(r"^\s*f(['\"{]{1,3})", r'\1', template, count=1)
    if '{' not in template: return template

    template = fix_latex_syntax(template)
    
    def is_evaluable(expr):
        expr = expr.strip()
        if not expr:
            return False
        forbidden = re.search(r';', expr) # expression with ; are not allowed
        assignment = re.search(r'(?<![=!<>])=(?![=])', expr) # assignment = not allowed, but ==, >+, etc are valid

        if forbidden or assignment:
            return False

        # Delete the content of the strings to avoid polluting the extraction of identifiers
        expr = re.sub(r"(['\"])(?:(?=(\\?))\2.)*?\1", "", expr)

        identifiers = set(re.findall(r'\b[a-zA-Z_]\w*\b', expr))
        allowed = set(var_names) | {
            'np', 'math', 'abs', 'int', 'float', 'str', 'pow',
            'round', 'min', 'max', 'sum', 'len', 'bool',  'if', 'else',
        }

        # autoriser attributs np.xxx et math.xxx
        attr_calls = re.findall(r'\b(np|math)\.([a-zA-Z_]\w*)', expr)

        for base, attr in attr_calls:
            if base == 'np' and attr not in dir(np):
                return False
            if base == 'math' and attr not in dir(math):
                return False

        # retirer les attributs du test simple
        identifiers -= {attr for _, attr in attr_calls}

        return identifiers <= allowed



    def process(text, in_math):
        result, i = [], 0
        while i < len(text):
            c = text[i]
            if c == '{' and i+1 < len(text) and text[i+1] == '{':
                result.append('{'); i += 2                          # {{ → literal {
            elif c == '}' and i+1 < len(text) and text[i+1] == '}':
                result.append('}'); i += 2                          # }} → literal }
            elif c == '{':
                # Find matching closing brace (depth-first, handles nesting)
                depth, j = 1, i+1
                while j < len(text) and depth:
                    depth += (text[j] == '{') - (text[j] == '}'); j += 1
                if depth != 0:
                    # No matching brace (split across math segment): leave intact
                    result.append('{'); i += 1
                    continue
                expr = text[i+1:j-1]
                if is_evaluable(expr):
                    if not in_math:
                        # Outside math: bare \py{expr}
                        result.append(f'\\py{{{expr}}}')
                    else:
                        # Inside math: wrap in braces if attached to LaTeX operator
                        last = ''.join(result)
                        standalone = not last or last[-1] in ' \t\n'
                        if standalone:
                            result.append(f'\\py{{{expr}}}')
                        else:
                            result.append('{' + f'\\py{{{expr}}}' + '}')
                else:
                    # Unknown: leave intact
                    result.append('{' + expr + '}')
                i = j
            else:
                result.append(c); i += 1
        return ''.join(result)

    # Split on $$...$$ (display) then $...$ (inline).
    # Capturing group keeps delimiters in the parts list.
    # Even indices → outside math, odd indices → math segment (delimiters included).
    parts = re.split(r'(?<!\\)(\$\$.+?\$\$|\$[^\$]+?\$)', template, flags=re.DOTALL)
    return ''.join(
        process(p, False) if i % 2 == 0
        else process(p[2:-2], True).join(('$$', '$$')) if p.startswith('$$')
        else process(p[1:-1], True).join(('$', '$'))
        for i, p in enumerate(parts)
    )


def process_braces(text):
    # 1. \py{...} -> |!...!|
    text = re.sub(r'\\pyc?\{(.*?)\}', r'|!\1!|', text)
    # or 
    # import regex # pip install regex
    # pattern = r'\\pyc?\{(?:[^{}]|(?R))*\}' and text = regex.sub(pattern, r'|!\1!|', text, flags=regex.VERBOSE)
    # 2. Double all remaining braces
    text = text.replace('{', '{{').replace('}', '}}')
    # 3. |!...!| -> {...}
    return text.replace('|!', '{').replace('!|', '}')

def evaluate_fstring(template, context):

    if template.replace(' ','').replace('{','').replace('}','').lower() == "true": return 'True'
    if template.replace(' ','').replace('{','').replace('}','').lower()  == "false": return 'False'

    var_names = list(context.keys())
    template = _template_to_py(template, var_names)
    template = process_braces(template)
    try:
        val = eval("f" + repr(template), safe_globals, context).strip("'").strip('"')
        return val
    except Exception as e:
        print("Error evaluating" + repr(template), e )
        return '[‼️ f-string evaluation error] ' + template
    


def safe_eval(expr):
    """
    Evaluate expression in a restricted namespace.
    Only rng, numpy and pandas are allowed.
    """
    return eval(expr, {"__builtins__": {}}, {"rng": rng, "np": np, "pd": pd})

def evaluate_text(text, context):
    #return html.escape(evaluate_fstring(text, context)).strip("'").strip('"')
    # Do not remember why I did put html.escape before..
    return evaluate_fstring(text, context).strip("'").strip('"')


def markdown_with_latex_to_html(text, mathml=True):
    import markdown
    from latex2mathml.converter import convert

    patterns = [
        r'\$\$(.+?)\$\$',
        r'\\\[(.+?)\\\]',
        r'\\\((.+?)\\\)',
        r'\$(.+?)\$',
    ]

    placeholders = {}
    counter = [0]

    def replace(match):
        if mathml:
            try:
                return convert(match.group(1))
            except:
                return match.group(0)
        else:
            key = f"LATEX{counter[0]}PLACEHOLDER"
            placeholders[key] = match.group(0)
            counter[0] += 1
            return key

    protected = text
    for pattern in patterns:
        protected = re.sub(pattern, replace, protected, flags=re.DOTALL)

    html = markdown.markdown(protected, extensions=['tables', 'fenced_code', 'nl2br'])

    if not mathml:
        for key, val in placeholders.items():
            html = html.replace(key, val)

    return html


def processPropositions(p, q_type, context):
    v_exp = p.get('expected', '') 
    v_prop = p.get('proposition', '')
    v_rep = p.get('answer', '')
    v_tip = p.get('tip', '')
    v_lab = evaluate_text(p.get('label', ''), context)
    if 'template' in q_type:
        if not '{' in v_exp: v_exp = f'{{ {v_exp} }}'
        v_exp = evaluate_fstring(v_exp, context)
        v_exp = v_exp.strip().strip("'").strip('"')
        v_prop = evaluate_text(v_prop, context)
        v_rep = evaluate_text(v_rep, context)
        v_tip = evaluate_text(v_tip, context)
        if "numeric" in q_type:
            v_exp = float(v_exp) # to extend later with type checking
        else:
            v_exp = v_exp == 'True'
    return v_prop, v_exp, v_rep, v_lab, v_tip

#!/usr/bin/env python3
"""
LabQuiz YAML → Moodle XML converter
=====================================
Supported LabQuiz question types:
  - mcq              → Moodle multichoice
  - numeric (1 prop) → Moodle numerical  (multiple <answer> with fraction + tolerance)
  - numeric (n prop) → Moodle cloze      ({1:NUMERICAL:=val:tol} embedded in questiontext)
  - mcq-template     → multichoice with [variable] placeholders + warning
  - numeric-template → numerical/cloze   with [variable] placeholders + warning

Score mapping (identical logic to labquiz_to_gift.py):
  max_score = sum of correctAnswerPoints values for all true-positive propositions (default correctAnswerPoints = 1).
  Each answer's fraction = correctAnswerPoints/max_score  (correct) or −incorrectAnswerPoints/max_score (incorrect).
  The default incorrectAnswerPoints for a false-positive is taken from DEFAULT_WEIGHTS[(True,False)] = 1,
  but can be overridden per-proposition with the 'incorrectAnswerPoints' key, or globally via the
  weights parameter of labquiz_to_moodle_xml().

Moodle XML notes (from the reference export):
  - <answer fraction="..."> accepts decimal percentages (e.g. 66.66667, -33.33333).
  - <single>true</single> is used for both single- and multi-answer MCQ in Moodle;
    Moodle infers single vs. multi from whether fractions sum to 100.
  - Cloze questions embed {1:NUMERICAL:=val:tol} directly in <questiontext>.
  - Text format is "html" by default; "markdown" is also supported and preserved
    when the source text contains markdown markers.

Usage:
    python labquiz_to_moodle_xml.py input.yaml output.xml
    python labquiz_to_moodle_xml.py input.yaml            # write to stdout
"""

import sys
import re
import xml.etree.ElementTree as ET
import yaml
from typing import Optional


# ---------------------------------------------------------------------------
# Default LabQuiz weight matrix — mirrors calculate_quiz_score() defaults
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    (True,  True):   1,   # True Positive  → +bonus  (default 1)
    (True,  False): -1,   # False Positive → -malus  (default 1)
    (False, True):   0,   # False Negative → -malus  (default 0)
    (False, False):  0,   # True Negative  → +bonus  (default 0)
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_max_score(propositions: list, weights: dict) -> float:
    """Sum of correctAnswerPointses for true-positive propositions (= maximum achievable score)."""
    total = 0.0
    for prop in propositions:
        expected = str(prop.get("expected", "false")).lower() == "true"
        if expected:
            total += float(prop.get("correctAnswerPoints", weights[(True, True)]))
        else:
            total += float(prop.get("correctAnswerPoints", weights[(False, False)]))
    return total if total > 0 else 1.0


def fraction_str(value: float) -> str:
    """Format a score fraction as a Moodle XML percentage string (up to 5 d.p.)."""
    pct = round(value * 100, 5)
    s = f"{pct:.5f}".rstrip("0").rstrip(".")
    return s


def xml_text(parent: ET.Element, tag: str, text: str,
             fmt: str = "html", cdata: bool = True) -> ET.Element:
    """
    Append  <tag format="fmt"><text>…</text></tag>  to parent.
    For CDATA content we store a unique token in .text; after ET serialises
    the tree, we replace those tokens with <![CDATA[…]]> sections.
    """
    el = ET.SubElement(parent, tag, format=fmt)
    t  = ET.SubElement(el, "text")
    if cdata and text:
        token = _cdata_token(text)
        t.text = token
    else:
        t.text = text or ""
    return el


def plain_text(parent: ET.Element, tag: str, text: str) -> ET.Element:
    """Append <tag><text>…</text></tag> without a format attribute."""
    el = ET.SubElement(parent, tag)
    ET.SubElement(el, "text").text = text or ""
    return el


def cdata_answer(ans: ET.Element, html: str) -> None:
    """Set the <text> child of an <answer> with CDATA content."""
    t = ET.SubElement(ans, "text")
    t.text = _cdata_token(html) if html else ""


def make_feedback(parent: ET.Element, text: str, fmt: str = "html") -> None:
    """Append <feedback format="fmt"><text>…</text></feedback>."""
    fb = ET.SubElement(parent, "feedback", format=fmt)
    t  = ET.SubElement(fb, "text")
    t.text = _cdata_token(text) if text else ""


# Registry mapping token → raw HTML (populated by _cdata_token)
_CDATA_REGISTRY: dict = {}


def _cdata_token(html: str) -> str:
    """
    Register raw HTML content and return an XML-safe token that ET can store
    in a .text attribute without escaping anything meaningful.
    The token contains only alphanumerics and underscores.
    """
    import uuid
    token = f"CDATATK_{uuid.uuid4().hex}"
    _CDATA_REGISTRY[token] = html
    return token


def resolve_template(text: str, variables: dict) -> str:
    """Replace {var} placeholders with [var] markers for template questions."""
    if not isinstance(text, str):
        return str(text)
    def replacer(m):
        name = m.group(1).strip()
        return f"[{name}]" if name in variables else m.group(0)
    return re.sub(r"\{(\w+)\}", replacer, text)


def evaluate_expected(expr: str) -> str:
    """Convert a template expected-value expression to a readable placeholder."""
    if not isinstance(expr, str):
        return str(expr)
    fstring = re.match(r"^f['\"](.+)['\"]$", expr.strip())
    if fstring:
        inner = fstring.group(1)
        formula = re.match(r"\{([^:}]+)", inner)
        return f"[{formula.group(1).strip()}]" if formula else f"[{inner}]"
    return f"[{expr.strip().strip('{}') }]"


def detect_format(text: str) -> str:
    """Heuristically detect whether text is HTML or Markdown."""
    if re.search(r"<[a-zA-Z][^>]*>", text):
        return "html"
    return "markdown"


def wrap_html(text: str, fmt: str) -> str:
    """Wrap plain text in <p> if the format is html and there is no HTML yet."""
    if fmt == "html" and not re.search(r"<[a-zA-Z]", text):
        return f"<p>{text}</p>"
    return text


# ---------------------------------------------------------------------------
# MCQ question builder
# ---------------------------------------------------------------------------

def build_multichoice(quiz_id: str, quiz: dict, root: ET.Element,
                      variables: dict = None, weights: dict = None,
                      is_template: bool = False) -> None:
    """Append a <question type="multichoice"> element to root."""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    question_text = quiz.get("question", "")
    if variables:
        question_text = resolve_template(question_text, variables)
    propositions = quiz.get("propositions", [])
    constraints  = quiz.get("constraints", [])

    fmt    = detect_format(question_text)
    q_text = wrap_html(question_text, fmt)

    q = ET.SubElement(root, "question", type="multichoice")
    plain_text(q, "name",
               f"{quiz_id} [TEMPLATE — review placeholders]" if is_template else quiz_id)
    xml_text(q, "questiontext", q_text, fmt=fmt)

    if constraints:
        warn = (f"WARNING: {len(constraints)} LabQuiz constraint(s) "
                "(XOR/SAME/IMPLY/IMPLYFALSE) could not be converted.")
        xml_text(q, "generalfeedback", f"<p>{warn}</p>", fmt="html")
    else:
        xml_text(q, "generalfeedback", "", fmt="html", cdata=False)

    ET.SubElement(q, "defaultgrade").text = "1.0000000"
    ET.SubElement(q, "penalty").text      = "0.3333333"
    ET.SubElement(q, "hidden").text       = "0"
    ET.SubElement(q, "idnumber")

    correct_count = sum(
        1 for p in propositions
        if str(p.get("expected", "false")).lower() == "true"
    )
    ET.SubElement(q, "single").text          = "true" if correct_count <= 1 else "false"
    ET.SubElement(q, "shuffleanswers").text  = "true"
    ET.SubElement(q, "answernumbering").text = "abc"
    ET.SubElement(q, "showstandardinstruction").text = "0"

    xml_text(q, "correctfeedback",          "<p>Your answer is correct.</p>",            fmt="html")
    xml_text(q, "partiallycorrectfeedback", "<p>Your answer is partially correct.</p>",  fmt="html")
    xml_text(q, "incorrectfeedback",        "<p>Your answer is incorrect.</p>",           fmt="html")
    ET.SubElement(q, "shownumcorrect")

    max_score = compute_max_score(propositions, weights)

    for prop in propositions:
        prop_text = prop.get("proposition", "")
        if variables:
            prop_text = resolve_template(prop_text, variables)

        expected = str(prop.get("expected", "false")).lower() == "true"

        if expected:
            correctAnswerPoints = float(prop.get("correctAnswerPoints", weights[(True, True)]))
            frac  = correctAnswerPoints / max_score
        else:
            incorrectAnswerPoints = float(prop.get("incorrectAnswerPoints", abs(weights[(True, False)])))
            frac  = -incorrectAnswerPoints / max_score

        prop_fmt  = detect_format(prop_text)
        prop_html = wrap_html(prop_text, prop_fmt)

        ans = ET.SubElement(q, "answer", fraction=fraction_str(frac), format=prop_fmt)
        cdata_answer(ans, prop_html)

        feedback_parts = []
        if prop.get("answer"):
            feedback_parts.append(str(prop["answer"]))
        if prop.get("tip"):
            feedback_parts.append(f"Hint: {prop['tip']}")
        fb_text = " — ".join(feedback_parts)
        make_feedback(ans, f"<p>{fb_text}</p>" if fb_text else "")


# ---------------------------------------------------------------------------
# Numerical question builder  (single proposition → type="numerical")
# ---------------------------------------------------------------------------

def build_numerical(quiz_id: str, quiz: dict, root: ET.Element,
                    variables: dict = None, weights: dict = None,
                    is_template: bool = False) -> None:
    """Append a <question type="numerical"> element to root."""
    prop = quiz["propositions"][0]
    question_text = quiz.get("question", "")
    prop_label    = prop.get("proposition", "")

    if variables:
        question_text = resolve_template(question_text, variables)
        prop_label    = resolve_template(prop_label, variables)

    full_question = f"{question_text} ({prop_label})" if prop_label else question_text
    fmt = detect_format(full_question)
    q_text = wrap_html(full_question, fmt)

    expected     = prop.get("expected", 0)
    tolerance_rel = prop.get("tolerance", 0)
    tolerance_abs = prop.get("tolerance_abs", 0)

    if variables:
        expected = evaluate_expected(str(expected))

    try:
        exp_val = float(expected)
        tol     = max(float(tolerance_abs), float(tolerance_rel) * abs(exp_val))
        exp_str = str(exp_val)
        tol_str = str(tol)
        formula_ref = None
    except (ValueError, TypeError):
        exp_str = '*'
        tol_str = str(tolerance_abs)
        formula_ref = f"{prop_label} = {expected}"

    general_fb = (
        "<p>Template formula for reference (also add :tolerance if needed):<br/>" +
        "{formula_ref}<br/>".format(formula_ref=formula_ref) + "</p>"
    ) if formula_ref else ""

    feedback_parts = []
    if prop.get("answer"):
        feedback_parts.append(str(prop["answer"]))
    if prop.get("tip"):
        feedback_parts.append(f"Hint: {prop['tip']}")
    fb_text = " — ".join(feedback_parts)

    q = ET.SubElement(root, "question", type="numerical")
    plain_text(q, "name", f"{quiz_id} [TEMPLATE — review]" if is_template else quiz_id)
    xml_text(q, "questiontext", q_text, fmt=fmt)
    xml_text(q, "generalfeedback", general_fb, fmt="html", cdata=bool(general_fb))
    ET.SubElement(q, "defaultgrade").text = "1.0000000"
    ET.SubElement(q, "penalty").text      = "0.3333333"
    ET.SubElement(q, "hidden").text       = "0"
    ET.SubElement(q, "idnumber")

    ans = ET.SubElement(q, "answer", fraction="100", format="moodle_auto_format")
    ET.SubElement(ans, "text").text = exp_str
    make_feedback(ans, f"<p>{fb_text}</p>" if fb_text else "")
    ET.SubElement(ans, "tolerance").text = tol_str

    ET.SubElement(q, "unitgradingtype").text = "0"
    ET.SubElement(q, "unitpenalty").text = "0.1000000"
    ET.SubElement(q, "showunits").text = "3"
    ET.SubElement(q, "unitsleft").text = "0"


# ---------------------------------------------------------------------------
# Cloze question builder  (multiple propositions → type="cloze")
# ---------------------------------------------------------------------------

def build_cloze(quiz_id: str, quiz: dict, root: ET.Element,
                variables: dict = None, is_template: bool = False) -> None:
    """
    Append a <question type="cloze"> element to root.
    Each proposition becomes a {1:NUMERICAL:=val:tol} placeholder embedded in
    an HTML list inside <questiontext>.
    """
    question_text = quiz.get("question", "")
    if variables:
        question_text = resolve_template(question_text, variables)

    # Build an HTML list with one NUMERICAL placeholder per proposition
    items        = []
    formula_refs = []
    for prop in quiz["propositions"]:
        prop_label = prop.get("proposition", "")
        if variables:
            prop_label = resolve_template(prop_label, variables)

        expected     = prop.get("expected", 0)
        tolerance_rel = prop.get("tolerance", 0.01)
        tolerance_abs = prop.get("tolerance_abs", 0)

        if variables:
            expected = evaluate_expected(str(expected))

        try:
            exp_val = float(expected)
            tol     = max(float(tolerance_abs), float(tolerance_rel) * abs(exp_val))
            placeholder = f"{{1:NUMERICAL:={exp_val}:{tol}}}"
            formula_refs.append(None)
        except (ValueError, TypeError):
            placeholder  = "{1:NUMERICAL:=*}"
            formula_refs.append(f"{prop_label} = {expected}")

        items.append(f"<li>{prop_label} : {placeholder}</li>")

    formulas = [r for r in formula_refs if r]
    general_fb = (
        "<p>Template formulas for reference (also add :tolerance if needed):<br/>" +
        "<br/>".join(formulas) + "</p>"
    ) if formulas else ""

    cloze_html = (
        f"<p>{question_text}</p>\n"
        "<ul>\n" + "\n".join(items) + "\n</ul>"
    )

    q = ET.SubElement(root, "question", type="cloze")
    plain_text(q, "name", f"{quiz_id} [TEMPLATE — review - insert formulas and tolerances]" if is_template else quiz_id)
    xml_text(q, "questiontext", cloze_html, fmt="html")
    xml_text(q, "generalfeedback", general_fb, fmt="html", cdata=bool(general_fb))
    ET.SubElement(q, "defaultgrade").text = "1.0000000"
    ET.SubElement(q, "penalty").text = "0.3333333"
    ET.SubElement(q, "hidden").text = "0"
    ET.SubElement(q, "idnumber")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def convert_quiz(quiz_id: str, quiz: dict, root: ET.Element, weights: dict = None) -> Optional[str]:
    """
    Convert one LabQuiz question and append it to the XML root.
    Returns a warning string if something was skipped or approximated, else None.
    """
    qtype     = quiz.get("type", "mcq").lower()
    variables = quiz.get("variables", {})
    is_tmpl   = qtype.endswith("-template")
    base_type = qtype.replace("-template", "").replace("qcm", "mcq")

    if base_type == "mcq":
        build_multichoice(quiz_id, quiz, root,
                          variables=variables if is_tmpl else None,
                          weights=weights,
                          is_template=is_tmpl)
    elif base_type == "numeric":
        props = quiz.get("propositions", [])
        if len(props) == 1:
            build_numerical(quiz_id, quiz, root,
                            variables=variables if is_tmpl else None,
                            weights=weights,
                            is_template=is_tmpl)
        else:
            build_cloze(quiz_id, quiz, root,
                        variables=variables if is_tmpl else None,
                        is_template=is_tmpl)
    else:
        return f"SKIPPED: {quiz_id} — unknown type '{qtype}'"

    if is_tmpl:
        return (f"WARNING: {quiz_id} is a template question — "
                "variables replaced by [name] placeholders, review before import.")
    return None


# ---------------------------------------------------------------------------
# Full file conversion
# ---------------------------------------------------------------------------

def labquiz_to_moodle_xml(content,
                           category: str = None,
                           weights: dict = None) -> str:
    if isinstance(content, dict):
        data = content
    else:
        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            raise ValueError("YAML file does not contain a valid dictionary.")

    title = data.get("title", "")
    cat   = category or title or "LabQuiz import"

    root = ET.Element("quiz")

    # Category block
    cat_q  = ET.SubElement(root, "question", type="category")
    cat_el = ET.SubElement(cat_q, "category")
    ET.SubElement(cat_el, "text").text = f"$course$/top/{cat}"
    info   = ET.SubElement(cat_q, "info", format="moodle_auto_format")
    ET.SubElement(info, "text").text = f"Imported from LabQuiz — {cat}"
    ET.SubElement(cat_q, "idnumber")

    warnings = []
    for key, value in data.items():
        if key == "title" or not isinstance(value, dict):
            continue
        w = convert_quiz(key, value, root, weights=weights)
        if w:
            warnings.append(w)

    # Pretty-print the tree in-place (Python 3.9+)
    ET.indent(root, space="  ")

    # Serialise — tokens in .text are alphanumeric so ET won't escape them.
    raw = ET.tostring(root, encoding="unicode", xml_declaration=False)

    # Replace each token with the corresponding <![CDATA[…]]> section.
    for token, html in _CDATA_REGISTRY.items():
        raw = raw.replace(token, f"<![CDATA[{html}]]>")
    _CDATA_REGISTRY.clear()

    pretty = f'<?xml version="1.0" encoding="UTF-8"?>\n{raw}\n'

    if warnings:
        warn_block = "\n".join(f"<!-- {w} -->" for w in warnings)
        pretty = pretty.replace("<quiz>", f"<quiz>\n{warn_block}", 1)

    return pretty


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    with open(input_path, "r", encoding="utf-8") as f:
        yaml_content = f.read()

    xml_content = labquiz_to_moodle_xml(yaml_content)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_content)
        print(f"Conversion complete: {output_path}")
    else:
        print(xml_content)


if __name__ == "__main__":
    main()

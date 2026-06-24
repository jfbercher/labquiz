#!/usr/bin/env python3
"""
Moodle XML → LabQuiz YAML converter
=====================================
Supported Moodle question types:
  - multichoice  → mcq
  - numerical    → numeric  (single proposition, multiple <answer> entries supported)
  - cloze        → numeric  (multiple propositions, {1:NUMERICAL:=val:tol} placeholders)

Unsupported types (skipped with a warning):
  - category, essay, shortanswer, matching, truefalse, description, ...

Score mapping (multichoice):
  Moodle <answer fraction="…"> percentages are converted to LabQuiz bonus/malus
  using the same denominator-inference logic as gift_to_labquiz.py:
    - A common integer denominator is inferred from all non-zero fractions.
    - bonus = round(pct/100 * denominator)  for correct answers (fraction > 0)
    - malus = round(|pct|/100 * denominator) for wrong answers  (fraction < 0)
    - Default bonus (1) and default malus (1) are omitted from the YAML output.

Score mapping (numerical):
  Only the highest-fraction <answer> is used as the expected value + tolerance.
  Partial-credit answers (fraction < 100) are noted as warnings.

Usage:
    python moodle_xml_to_labquiz.py input.xml output.yaml
    python moodle_xml_to_labquiz.py input.xml            # write to stdout
"""

import sys
import re
import xml.etree.ElementTree as ET
import yaml


# ---------------------------------------------------------------------------
# Reuse helpers from gift_to_labquiz
# (copied here to keep the file self-contained)
# ---------------------------------------------------------------------------

def clean_html(text: str) -> str:
    """Strip HTML tags, decode common entities, normalise whitespace."""
    if not text:
        return ""
    # Convert <br>, <p>, <li> to newlines for readability
    text = re.sub(r"<br\s*/?>",      "\n",  text, flags=re.IGNORECASE)
    text = re.sub(r"</p>",           "\n",  text, flags=re.IGNORECASE)
    text = re.sub(r"<p[^>]*>",       "",    text, flags=re.IGNORECASE)
    text = re.sub(r"<li[^>]*>",      "- ",  text, flags=re.IGNORECASE)
    text = re.sub(r"</li>",          "\n",  text, flags=re.IGNORECASE)
    text = re.sub(r"<ul[^>]*>|</ul>|<ol[^>]*>|</ol>", "", text, flags=re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode entities
    text = text.replace("&lt;",   "<")
    text = text.replace("&gt;",   ">")
    text = text.replace("&amp;",  "&")
    text = text.replace("&nbsp;", " ")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;",  "'")
    # Collapse excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_name(text: str, index: int) -> str:
    """Generate a valid LabQuiz identifier from question text."""
    slug = re.sub(r"[^a-zA-Z0-9]", "_", text[:30]).strip("_").lower()
    slug = re.sub(r"_+", "_", slug).strip("_")
    return f"{slug}_{index}" if slug else f"quiz_{index}"


def xml_name(name: str) -> str:
    """Sanitise a Moodle question name into a valid LabQuiz key."""
    slug = re.sub(r"[^a-zA-Z0-9_]", "_", name).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return slug or "quiz"


# ---------------------------------------------------------------------------
# Score inference (identical to gift_to_labquiz.py)
# ---------------------------------------------------------------------------

def infer_denominator(percentages: list) -> int:
    """
    Infer the smallest integer denominator D such that every percentage p
    satisfies round(p * D / 100) >= 1 AND the mapping is accurate (error < 0.5).
    """
    if not percentages:
        return 1
    for d in range(1, 21):
        ok = True
        for p in percentages:
            pts = p * d / 100
            if p > 0 and round(pts) < 1:
                ok = False
                break
            if abs(round(pts) - pts) >= 0.5:
                ok = False
                break
        if ok:
            return d
    return 1


def pct_to_points(pct: float, denominator: int) -> int:
    """Convert a Moodle fraction percentage to integer LabQuiz points."""
    return max(0, round(abs(pct) * denominator / 100))


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def get_text(element, tag: str, default: str = "") -> str:
    """Return the text of the first child <tag><text>…</text></tag>, or default."""
    child = element.find(tag)
    if child is None:
        return default
    t = child.find("text")
    if t is None or not t.text:
        return default
    return t.text.strip()


def get_format(element, tag: str) -> str:
    """Return the format attribute of a child tag (html, markdown, …)."""
    child = element.find(tag)
    if child is None:
        return "html"
    return child.get("format", "html")


def question_text_clean(q_el: ET.Element) -> str:
    """
    Extract and clean the question text from <questiontext>.
    Preserves the text as-is for markdown format; strips HTML for html format.
    """
    fmt  = get_format(q_el, "questiontext")
    raw  = get_text(q_el,  "questiontext")
    if fmt == "markdown":
        return raw.strip()
    return clean_html(raw)


# ---------------------------------------------------------------------------
# Multichoice builder
# ---------------------------------------------------------------------------

def build_multichoice(q_el: ET.Element, quiz_name: str) -> tuple:
    """
    Convert a <question type="multichoice"> to an mcq LabQuiz question.
    Returns (quiz_dict, warning_or_None).
    """
    question = question_text_clean(q_el)
    answers  = q_el.findall("answer")

    if not answers:
        return None, f"Question '{quiz_name}': no answers found."

    # Collect options with their fractions
    options = []
    for ans in answers:
        fraction = float(ans.get("fraction", "0"))
        fmt      = ans.get("format", "html")
        raw_text = ""
        t = ans.find("text")
        if t is not None and t.text:
            raw_text = t.text.strip()
        text = raw_text if fmt == "markdown" else clean_html(raw_text)

        fb_raw = get_text(ans, "feedback")
        fb_fmt = get_format(ans, "feedback")
        feedback = fb_raw if fb_fmt == "markdown" else clean_html(fb_raw)

        if not text:
            continue
        options.append({"fraction": fraction, "text": text, "feedback": feedback})

    # Infer denominator from non-zero fractions
    nonzero = [abs(o["fraction"]) for o in options if o["fraction"] != 0.0]
    denominator = infer_denominator(nonzero)

    propositions = []
    approximate  = False
    counter      = 1

    for opt in options:
        frac   = opt["fraction"]
        points = pct_to_points(frac, denominator)
        is_correct = frac > 0

        # Accuracy check
        if denominator > 0 and frac != 0.0:
            reconstructed = round(points / denominator * 100, 5)
            if abs(reconstructed - abs(frac)) > 0.6:
                approximate = True

        prop = {
            "proposition": opt["text"],
            "label":       f"{quiz_name}_p{counter}",
            "type":        "bool",
            "expected":    is_correct,
        }

        if is_correct:
            #if points != 1:
            prop["bonus"] = points
            prop["malus"] = 0
        else:
            prop["malus"] = points
            prop["bonus"] = 0

        if opt["feedback"]:
            prop["answer"] = opt["feedback"]

        propositions.append(prop)
        counter += 1

    warning = (
        f"Question '{quiz_name}': score mapping is approximate "
        "(Moodle fractions did not convert to clean integer bonus/malus values)."
        if approximate else None
    )

    return (
        {"question": question, "type": "mcq", "propositions": propositions},
        warning,
    )


# ---------------------------------------------------------------------------
# Numerical builder
# ---------------------------------------------------------------------------

def build_numerical(q_el: ET.Element, quiz_name: str) -> tuple:
    """
    Convert a <question type="numerical"> to a numeric LabQuiz question.
    The highest-fraction <answer> becomes the single proposition.
    Other answers (partial credit) are noted as warnings.
    """
    question = question_text_clean(q_el)
    answers  = q_el.findall("answer")

    if not answers:
        return None, f"Question '{quiz_name}': no answers found."

    # Sort by fraction descending; take the best answer as the expected value
    def answer_fraction(a):
        return float(a.get("fraction", "0"))

    answers_sorted = sorted(answers, key=answer_fraction, reverse=True)
    best = answers_sorted[0]

    fraction = float(best.get("fraction", "100"))
    val_el   = best.find("text")
    val_str  = (val_el.text or "").strip() if val_el is not None else ""
    tol_el   = best.find("tolerance")
    tol_str  = (tol_el.text or "0").strip() if tol_el is not None else "0"

    fb_raw = get_text(best, "feedback")
    fb_fmt = get_format(best, "feedback")
    feedback = fb_raw if fb_fmt == "markdown" else clean_html(fb_raw)

    try:
        exp = float(val_str)
        expected = int(exp) if exp == int(exp) else exp
    except ValueError:
        expected = val_str

    try:
        tolerance_abs = float(tol_str)
    except ValueError:
        tolerance_abs = 0.0

    # Warn if there are additional partial-credit answers we can't represent
    warning = None
    if len(answers_sorted) > 1:
        warning = (
            f"Question '{quiz_name}': {len(answers_sorted) - 1} partial-credit "
            "answer(s) ignored — LabQuiz numeric supports only a single expected value."
        )

    # Extract label from parenthesised suffix in question text, if any
    q_clean = question
    paren_m = re.search(r"\(([^)]+)\)\s*$", q_clean)
    if paren_m:
        prop_label = paren_m.group(1).strip()
        q_clean    = q_clean[:paren_m.start()].strip()
    else:
        prop_label = "Answer"

    prop = {
        "proposition":  prop_label,
        "label":        f"{quiz_name}_v1",
        "type":         "int" if isinstance(expected, int) else "float",
        "expected":     expected,
        "tolerance_abs": tolerance_abs,
        "tip":          "Enter the value",
    }
    if feedback:
        prop["answer"] = feedback

    return (
        {"question": q_clean, "type": "numeric", "propositions": [prop]},
        warning,
    )


# ---------------------------------------------------------------------------
# Cloze builder  (reuses the GIFT Cloze parser from gift_to_labquiz)
# ---------------------------------------------------------------------------

def build_cloze(q_el: ET.Element, quiz_name: str) -> tuple:
    """
    Convert a <question type="cloze"> to a numeric LabQuiz question.
    The {1:NUMERICAL:=val:tol} placeholders embedded in <questiontext> are
    extracted with the same regex used by the GIFT/XML converter.
    """
    fmt = get_format(q_el, "questiontext")
    raw = get_text(q_el, "questiontext")

    # For HTML format, decode entities but keep the NUMERICAL placeholders intact
    if fmt != "markdown":
        # Decode only entities, do not strip tags yet (placeholders may be inline)
        raw = raw.replace("&lt;",  "<").replace("&gt;",  ">") \
                 .replace("&amp;", "&").replace("&nbsp;", " ")

    pattern = re.compile(
        r"\{\d*\s*:\s*NUMERICAL\s*:\s*=\s*(-?[\d.eE+\-\*]+)\s*"
        r"(?::\s*(-?[\d.]+))?\s*(?:#([^}]*))?\s*\}",
        re.IGNORECASE,
    )

    propositions = []
    counter      = 1
    last_end     = 0

    for m in pattern.finditer(raw):
        val_str = m.group(1).strip()
        tol_str = (m.group(2) or "0").strip()
        fb_raw  = (m.group(3) or "").strip()
        feedback = clean_html(fb_raw) if fmt != "markdown" else fb_raw

        try:
            exp = float(val_str)
            expected = int(exp) if exp == int(exp) else exp
        except ValueError:
            expected = val_str
        try:
            tolerance_abs = float(tol_str)
        except ValueError:
            tolerance_abs = 0.0

        # Label = text just before the placeholder on the same line
        before    = raw[last_end:m.start()]
        last_line = before.split("\n")[-1]
        prop_text = clean_html(last_line).strip() or f"Value {counter}"
        # Remove leading list markers and trailing punctuation
        prop_text = re.sub(r"^[-*•]\s*", "", prop_text).rstrip(": -–").strip() or f"Value {counter}"

        prop = {
            "proposition":   prop_text,
            "label":         f"{quiz_name}_v{counter}",
            "type":          "int" if isinstance(expected, int) else "float",
            "expected":      expected,
            "tolerance_abs": tolerance_abs,
            "tip":           "Enter the value",
        }
        if feedback:
            prop["answer"] = feedback

        propositions.append(prop)
        counter  += 1
        last_end  = m.end()

    if not propositions:
        return None, f"Question '{quiz_name}': no NUMERICAL placeholders found in cloze."

    # Question text = everything before the first placeholder, stripped of HTML
    preamble = raw[:raw.index("{")]  if "{" in raw else raw
    q_text   = clean_html(preamble).split("\n")[0].strip()

    return (
        {"question": q_text, "type": "numeric", "propositions": propositions},
        None,
    )


# ---------------------------------------------------------------------------
# Full file conversion
# ---------------------------------------------------------------------------

SUPPORTED_TYPES = {"multichoice", "numerical", "cloze"}
IGNORED_TYPES   = {"category", "essay", "shortanswer", "matching",
                   "truefalse", "description", "calculated"}


def moodle_xml_to_labquiz(xml_content: str) -> tuple:
    """
    Parse a Moodle XML export and convert all supported questions to LabQuiz YAML.
    Returns (quiz_dict, warnings_list).
    """
    root = ET.fromstring(xml_content)

    result   = {}
    warnings = []
    index    = 1

    for q_el in root.findall("question"):
        qtype = q_el.get("type", "").lower()

        if qtype in IGNORED_TYPES:
            continue
        if qtype not in SUPPORTED_TYPES:
            warnings.append(f"Question #{index}: unsupported type '{qtype}' — skipped.")
            index += 1
            continue

        # Determine the quiz key
        name_text = get_text(q_el, "name")
        if name_text:
            quiz_name = xml_name(name_text)
        else:
            q_raw = get_text(q_el, "questiontext")
            quiz_name = safe_name(clean_html(q_raw), index)

        # Deduplicate
        base   = quiz_name
        suffix = 1
        while quiz_name in result:
            quiz_name = f"{base}_{suffix}"
            suffix += 1

        # Dispatch
        if qtype == "multichoice":
            data, warn = build_multichoice(q_el, quiz_name)
        elif qtype == "numerical":
            data, warn = build_numerical(q_el, quiz_name)
        elif qtype == "cloze":
            data, warn = build_cloze(q_el, quiz_name)
        else:
            data, warn = None, None

        if warn:
            warnings.append(warn)
        if data:
            result[quiz_name] = data

        index += 1

    return result, warnings


# ---------------------------------------------------------------------------
# YAML serialisation
# ---------------------------------------------------------------------------

def to_yaml(data: dict) -> str:
    return yaml.dump(
        data,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        indent=2,
        width=120,
    )


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
        xml_content = f.read()

    quizzes, warnings = moodle_xml_to_labquiz(xml_content)

    header = (
        "# LabQuiz YAML file generated from Moodle XML\n"
        "# Review labels, feedback, and tolerances before use.\n"
        "#\n"
        "# Score mapping: Moodle fractions → LabQuiz bonus/malus\n"
        "#   A common denominator is inferred from the fractions in each question.\n"
        "#   bonus = round(fraction/100 * denominator)  for correct answers\n"
        "#   malus = round(|fraction|/100 * denominator) for wrong answers\n"
        "#   Default bonus (1) and default malus (1) are omitted from the output.\n"
    )
    if warnings:
        header += "#\n# Warnings:\n"
        for w in warnings:
            header += f"#   {w}\n"
    header += "\n"

    yaml_output = header + to_yaml(quizzes)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(yaml_output)
        print(f"Conversion complete: {output_path}")
        if warnings:
            print(f"  {len(warnings)} warning(s) — see header of the YAML file.")
    else:
        print(yaml_output)


if __name__ == "__main__":
    main()

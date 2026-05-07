#!/usr/bin/env python3

# A small, basic and incomplete utility for converting AMC-LaTeX to the LabQuiz YAML file format.
# This strips all comments, 
# only works for english form (choices/correctchoic/wrongchoice, etc)



import re
import yaml
import argparse
import sys


# -----------------------------
# Nettoyage
# -----------------------------
def clean_latex(text):
    text = re.sub(r"%.*", "", text)
    return text.strip()


# -----------------------------
# Extraction des éléments
# -----------------------------
def extract_elements(tex):
    pattern = r"""
    \\element\{(?P<category>.*?)\}
    \s*\{
    (?P<content>.*?\\end\{(?:question|questionmult)\})
    \s*\}
    """

    matches = list(re.finditer(pattern, tex, re.DOTALL | re.VERBOSE))

    elements = []
    for m in matches:
        category = m.group("category").strip()
        content = m.group("content").strip()
        elements.append((category, content))

    return elements


# -----------------------------
# Extraction des choix
# -----------------------------
def extract_choices(block):
    pattern = r"\\(correctchoice|wrongchoice)\{(.*?)\}"
    matches = re.findall(pattern, block, re.DOTALL)

    results = []
    for kind, text in matches:
        results.append((kind == "correctchoice", text.strip()))

    return results


# -----------------------------
# Parsing d'une question
# -----------------------------
def parse_question_block(category, content, quiz_id):

    # Extraire label
    label_match = re.search(r"\\label\{(.*?)\}", content)
    label = label_match.group(1).strip() if label_match else f"q{quiz_id}"

    # Supprimer le label du contenu
    content = re.sub(r"\\label\{.*?\}", "", content)

    # question + identifiant questionmult
    q_match = re.search(
        r"\\begin\{(?:question|questionmult)\}\{(.*?)\}(.*?)\\begin\{choices\}",
        content,
        re.DOTALL
    )

    if not q_match:
        return None

    question_text = q_match.group(2).strip()
    question_text = re.sub(r"\s+", " ", question_text)

    # bloc choix
    choices_block = re.search(
        r"\\begin\{choices\}(.*?)\\end\{choices\}",
        content,
        re.DOTALL
    )

    propositions = []

    if choices_block:
        matches = extract_choices(choices_block.group(1))

        for i, (is_correct, text) in enumerate(matches):
            propositions.append({
                "proposition": text,
                "label": f"p{i+1}",
                "expected": is_correct,
                "tip": "",
                "answer": ""
            })

    return {
        "question": question_text,
        "propositions": propositions,
        "type": "mcq",
        "label": label,
        "category": category,
        "tags": []
    }


# -----------------------------
# Conversion principale
# -----------------------------
def latex_to_yaml(tex):
    tex = clean_latex(tex)

    elements = extract_elements(tex)

    quizzes = {}
    quiz_id = 1

    for category, content in elements:
        parsed = parse_question_block(category, content, quiz_id)

        if parsed:
            quizzes[f"quiz{quiz_id}"] = parsed
            quiz_id += 1

    return yaml.dump(quizzes, sort_keys=False, allow_unicode=True)


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert LaTeX quiz file to YAML format"
    )

    parser.add_argument(
        "input",
        help="Input LaTeX file (.tex)"
    )


    parser.add_argument(
        "-o", "--output",
        help="Output file (default: stdout)"
    )

    args = parser.parse_args()

    # Lire fichier entrée
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            tex = f.read()
            #print("text read", tex)
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)

    # Conversion
    yaml_output = latex_to_yaml(tex)

    # Écriture sortie

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(yaml_output)
    else:
        print(yaml_output)

    print(f"✅ Conversion successful: {args.output}")


if __name__ == "__main__":
    main()
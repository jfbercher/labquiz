import yaml
import argparse
import sys
import shutil
import copy
from pathlib import Path

def convert_quiz_data_v1_to_v2(data):
    """
    Convert data from v1 to v2 format
    """
    new_data = copy.deepcopy(data)

    # Conversion logic: v1 -> v2
    for quiz_id, entry in new_data.items():
        if quiz_id == "title": continue
        propositions = entry.get("propositions") or []
        constraints = entry.get("constraints", [])
        quiz_type = entry.get("type", "mcq")
        
        # Normalize types
        if quiz_type == "qcm":  
            entry['type'] = "mcq"
        elif quiz_type == "qcm-template": 
            entry['type'] = "mcq-template"

        # Rename 'reponse' key to 'answer' in propositions
        for prop in propositions: 
            if "reponse" in prop:
                prop["answer"] = prop.pop("reponse")
        # Rename 'indices' key to 'indexes' in constraints
        for constraint in constraints:
            if "indices" in constraint:
                constraint["indexes"] = constraint.pop("indices")

    return new_data


def convert_quiz_v1_to_v2(input_path: str, output_path: str = None, skip_backup: bool = False):
    """
    Loads a YAML file, converts quiz format from v1 to v2, 
    and saves the result. Automatically creates a backup if overwriting.
    """
    path = Path(input_path)
    
    if not path.exists():
        print(f"Error: The file '{input_path}' was not found.")
        sys.exit(1)

    # Determine destination
    target_path = Path(output_path) if output_path else path
    is_overwriting = target_path.resolve() == path.resolve()

    # Automatic backup logic: only if overwriting and not skipped
    if is_overwriting and not skip_backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        try:
            shutil.copy2(path, backup_path)
            print(f"Safety backup created: {backup_path}")
        except Exception as e:
            print(f"Warning: Could not create backup: {e}")
            # Optional: sys.exit(1) here if you want to force backup success

    # Load data
    try:
        with open(path, "r", encoding="utf-8") as f:
            quiz_bank = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Error reading YAML: {e}")
        sys.exit(1)

    # Conversion logic: v1 -> v2
    for quiz_id, entry in quiz_bank.items():
        if quiz_id == "title": continue
        propositions = entry.get("propositions") or []
        quiz_type = entry.get("type", "mcq")
        
        # Normalize types
        if quiz_type == "qcm":  
            entry['type'] = "mcq"
        elif quiz_type == "qcm-template": 
            entry['type'] = "mcq-template"

        # Rename 'reponse' key to 'answer' in propositions
        for prop in propositions: 
            if "reponse" in prop:
                prop["answer"] = prop.pop("reponse")

    # Save processed data
    try:
        with open(target_path, "w", encoding="utf-8") as f:
            yaml.dump(quiz_bank, f, allow_unicode=True, sort_keys=False, indent=2)
        
        status = "overwritten" if is_overwriting else f"saved to '{target_path}'"
        print(f"Success: File {status}.")
    except Exception as e:
        print(f"Error writing to file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="CLI tool to convert Quiz YAML files (v1 to v2) with automatic backup."
    )
    
    parser.add_argument(
        "input", 
        help="Path to the source YAML file (v1)"
    )
    parser.add_argument(
        "output", 
        nargs="?", 
        help="Path to the destination file (optional, defaults to overwriting source)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable automatic backup when overwriting the source file"
    )

    args = parser.parse_args()
    convert_quiz_v1_to_v2(args.input, args.output, args.no_backup)

if __name__ == "__main__":
    main()
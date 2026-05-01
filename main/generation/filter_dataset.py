import os
import sys
import json
import argparse
import shutil
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from main.utils import load_json

def filter_dataset(input_dir: str, dry_run: bool = False, delete_discarded: bool = False):
    # Determine output directory
    # Assume we are running from project root or similar. 
    # We want to place the discard folder in 'synthetic_dataset/{folder_name}_discard'
    
    abs_input_dir = os.path.abspath(input_dir)
    folder_name = os.path.basename(os.path.normpath(abs_input_dir))
    
    # Locate synthetic_dataset root relative to this script or current working dir?
    # The user instruction says "move ... to a folder name synthetic_dataset/{original_name}_discard"
    # We will assume 'synthetic_dataset' is in the current working directory or we create it there.
    # However, if the input_dir is ALREADY inside synthetic_dataset, we might want to stay there.
    # Let's check if 'synthetic_dataset' exists in CWD.
    
    base_discard_path = "synthetic_dataset"
    discard_dir = None

    if not delete_discarded:
        if not os.path.exists(base_discard_path):
            # If running from somewhere else, maybe try to find it relative to input_dir?
            # Or just create it in CWD.
            os.makedirs(base_discard_path, exist_ok=True)
            
        discard_dir = os.path.join(base_discard_path, f"{folder_name}_discard")
        
        if not dry_run:
            os.makedirs(discard_dir, exist_ok=True)
            print(f"Discarded files will be moved to: {discard_dir}")
        else:
            print(f"Dry run: Discarded files would be moved to: {discard_dir}")
    else:
        print("Discarded files will be DELETED.")

    json_files = []
    for dirpath, _, files in os.walk(input_dir):
        for fn in files:
            if fn.endswith(".json"):
                json_files.append(os.path.join(dirpath, fn))
    
    print(f"Found {len(json_files)} files to check.")
    
    discard_count = 0
    kept_count = 0
    reasons_count = {
        "redundancy": 0,
        "realism": 0,
        "inducibility": 0,
        "ambiguity": 0
    }
    
    for file_path in tqdm(json_files, desc="Filtering"):
        try:
            data = load_json(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
        meta = data.get("meta", {})
        quality = meta.get("quality")
        
        # If no quality check exists, what to do?
        # User instruction implies filtering based on these values. 
        # If values are missing, we can't filter strictly. 
        # Assume we keep them or warn? 
        # Usually filtering implies we want high quality. If unchecked, maybe keep or skip?
        # Let's assume we skip filtering (keep file) if quality check is missing, 
        # or maybe we should only filter if quality check IS present.
        # But if the goal is to clean the dataset, unchecked files might be suspicious.
        # For now, if no quality data, we pass (keep).
        
        if not quality:
            kept_count += 1
            continue
            
        should_discard = False
        reason = []
        
        # Criteria 1: redundancy > 40
        redundancy = quality.get("redundancy")
        if redundancy is not None and redundancy > 40:
            should_discard = True
            reasons_count["redundancy"] += 1
            reason.append(f"redundancy={redundancy}")
            
        # Criteria 2: realism/inducibility < 80
        realism = quality.get("realism")
        inducibility = quality.get("prompt_inducibility")
        
        if realism is not None and realism < 80:
            should_discard = True
            reasons_count["realism"] += 1
            reason.append(f"realism={realism}")
            
        if inducibility is not None and inducibility < 80:
            should_discard = True
            reasons_count["inducibility"] += 1
            reason.append(f"inducibility={inducibility}")
            
        # Criteria 3: ambiguous is True
        ambiguity = quality.get("ambiguity")
        if ambiguity is True:
            should_discard = True
            reasons_count["ambiguity"] += 1
            reason.append("ambiguous=True")
            
        if should_discard:
            discard_count += 1
            # Calculate relative path to maintain structure inside discard folder
            rel_path = os.path.relpath(file_path, input_dir)
            
            if not dry_run:
                if delete_discarded:
                    os.remove(file_path)
                else:
                    dest_path = os.path.join(discard_dir, rel_path)
                    dest_folder = os.path.dirname(dest_path)
                    os.makedirs(dest_folder, exist_ok=True)
                    shutil.move(file_path, dest_path)
            
            # print(f"Discarding {os.path.basename(file_path)}: {', '.join(reason)}")
        else:
            kept_count += 1

    print(f"\nFiltering complete.")
    print(f"Kept: {kept_count}")
    print(f"Discarded: {discard_count}")
    print("Discard reasons:")
    for r, c in reasons_count.items():
        print(f"  - {r}: {c}")

    if not dry_run:
        if delete_discarded:
            print(f"Discarded files were deleted.")
        else:
            print(f"Discarded files moved to {discard_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter dataset based on quality metrics.")
    parser.add_argument("input_dir", help="Directory containing JSON problems")
    parser.add_argument("--dry-run", action="store_true", help="Simulate filtering without moving files")
    parser.add_argument("--delete", action="store_true", help="Delete discarded files instead of moving them")
    
    args = parser.parse_args()
    
    filter_dataset(args.input_dir, args.dry_run, args.delete)


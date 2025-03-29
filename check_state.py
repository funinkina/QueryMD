import os
import json
import time
from pathlib import Path
import toml
from embeddings_manager import remove_document_from_collection, process_file_for_embeddings

config = toml.load("config.toml")

DOCUMENTS_DIR = Path(config["files"]["markdown_directory"])
STATE_FILE = Path(config['files']['state_file'])

def load_previous_state(state_file_path):
    if state_file_path.exists():
        try:
            with open(state_file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: State file {state_file_path} is corrupted. Starting fresh.")
            return {}
    return {}

def save_current_state(state_data, state_file_path):
    with open(state_file_path, 'w') as f:
        json.dump(state_data, f, indent=4)

def is_markdown_file(file_path):
    extensions = ['.md', '.markdown']
    return file_path.suffix.lower() in extensions

def check_files_state():
    previous_state = load_previous_state(STATE_FILE)
    current_state = {}
    found_files = set()
    files_to_process = []
    files_to_remove = []
    print(f"Scanning directory for markdown files: {DOCUMENTS_DIR}")

    for root, _, filenames in os.walk(DOCUMENTS_DIR):
        for filename in filenames:
            file_path = Path(root) / filename

            if not is_markdown_file(file_path):
                continue

            abs_path_str = str(file_path.resolve())
            found_files.add(abs_path_str)

            try:
                current_mtime = os.path.getmtime(file_path)
                current_size = os.path.getsize(file_path)
                current_state[abs_path_str] = {
                    'mtime': current_mtime,
                    'size': current_size
                }

                if abs_path_str not in previous_state:
                    print(f"Detected new markdown file: {file_path}")
                    files_to_process.append(abs_path_str)
                elif (previous_state[abs_path_str]['mtime'] != current_mtime or previous_state[abs_path_str]['size'] != current_size):
                    print(f"Detected modified markdown file: {file_path}")
                    files_to_process.append(abs_path_str)

            except OSError as e:
                print(f"Error accessing file {file_path}: {e}. Skipping.")

    previous_files = {path for path in previous_state.keys() if Path(path).suffix.lower() in ['.md', '.markdown']}
    deleted_files = previous_files - found_files
    files_to_remove.extend(list(deleted_files))

    if deleted_files:
        print(f"Detected deleted markdown files: {', '.join(map(str, deleted_files))}")

    if not files_to_process and not files_to_remove:
        save_current_state(current_state, STATE_FILE)
        return False

    print("\nProcessing changes...")

    files_needing_removal_in_db = files_to_remove + files_to_process
    for file_path_str in files_needing_removal_in_db:
        if file_path_str in previous_state:
            print(f"  - Removing old embeddings for: {file_path_str}")
            relative_path = Path(file_path_str).relative_to(DOCUMENTS_DIR.resolve())
            remove_document_from_collection(str(relative_path))

    for file_path_str in files_to_process:
        print(f"  - Processing/Embedding: {file_path_str}")
        process_file_for_embeddings(file_path_str, DOCUMENTS_DIR)

    print("Processing complete.")
    save_current_state(current_state, STATE_FILE)
    return True

if __name__ == "__main__":
    if not DOCUMENTS_DIR.is_dir():
        print(f"Error: Document directory not found: {DOCUMENTS_DIR}")
    else:
        start_time = time.time()
        check_files_state()
        end_time = time.time()
        print(f"\nFile check and processing finished in {end_time - start_time:.2f} seconds.")

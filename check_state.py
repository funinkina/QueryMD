import os
import json
from pathlib import Path
import toml
from embeddings_manager import remove_document_from_collection, process_file_for_embeddings

config = toml.load("config.toml")

DOCUMENTS_DIR = Path(config["files"]["markdown_directory"]).resolve()
STATE_FILE = Path(config['files']['state_file']).resolve()

def load_previous_state(state_file_path):
    if state_file_path.exists():
        try:
            with open(state_file_path, 'r') as f:
                content = f.read()
                if not content:
                    print(f"Warning: State file {state_file_path} is empty. Starting fresh.")
                    return {}
                return json.loads(content)
        except json.JSONDecodeError:
            print(f"Warning: State file {state_file_path} is corrupted. Starting fresh.")
            return {}
        except Exception as e:
            print(f"Error loading state file {state_file_path}: {e}. Starting fresh.")
            return {}
    return {}

def save_current_state(state_data, state_file_path):
    try:
        state_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file_path, 'w') as f:
            json.dump(state_data, f, indent=4)
    except Exception as e:
        print(f"Error saving state file {state_file_path}: {e}")


def is_markdown_file(file_path):
    extensions = ['.md', '.markdown']
    return file_path.suffix.lower() in extensions if file_path.suffix else False

def check_files_state():
    if not DOCUMENTS_DIR.is_dir():
        print(f"Error: Document directory not found or is not a directory: {DOCUMENTS_DIR}")
        return False

    previous_state = load_previous_state(STATE_FILE)
    current_state = {}
    found_files = set()
    files_to_process = []
    files_to_remove = []
    changes_made = False

    print(f"Scanning directory for markdown files: {DOCUMENTS_DIR}")

    for root, _, filenames in os.walk(DOCUMENTS_DIR):
        for filename in filenames:
            file_path = Path(root) / filename

            if not is_markdown_file(file_path):
                continue

            abs_path_str = str(file_path.resolve())
            found_files.add(abs_path_str)

            try:
                if not file_path.is_file():
                    print(f"Warning: File disappeared during scan: {file_path}. Skipping.")
                    continue

                current_mtime = os.path.getmtime(file_path)
                current_size = os.path.getsize(file_path)
                current_state[abs_path_str] = {
                    'mtime': current_mtime,
                    'size': current_size
                }

                prev_file_state = previous_state.get(abs_path_str)
                if not prev_file_state:
                    # print(f"Detected new file: {file_path.relative_to(DOCUMENTS_DIR)}")
                    files_to_process.append(abs_path_str)
                elif (prev_file_state['mtime'] != current_mtime or prev_file_state['size'] != current_size):
                    # print(f"Detected modified file: {file_path.relative_to(DOCUMENTS_DIR)}")
                    files_to_process.append(abs_path_str)

            except OSError as e:
                print(f"Error accessing file {file_path}: {e}. Skipping.")
            except Exception as e:
                print(f"Unexpected error processing file {file_path}: {e}. Skipping.")

    previous_files_abs = set(previous_state.keys())
    deleted_files_abs = previous_files_abs - found_files
    files_to_remove.extend(list(deleted_files_abs))

    if deleted_files_abs:
        deleted_relative = [str(Path(p).relative_to(DOCUMENTS_DIR)) for p in deleted_files_abs]
        # print(f"Detected deleted files: {', '.join(deleted_relative)}")

    if not files_to_process and not files_to_remove:
        print("No changes detected in markdown files.")
        save_current_state(current_state, STATE_FILE)
        return False

    print("\nProcessing detected changes...")
    changes_made = True
    files_needing_removal_in_db = files_to_remove + files_to_process

    if files_needing_removal_in_db:
        # print("  - Removing outdated/deleted document embeddings...")
        relative_paths_to_remove = set()
        for file_path_str in files_needing_removal_in_db:
            if file_path_str in previous_state:
                try:
                    relative_path = str(Path(file_path_str).relative_to(DOCUMENTS_DIR))
                    relative_paths_to_remove.add(relative_path)
                except ValueError:
                    print(f"Warning: Could not make {file_path_str} relative to {DOCUMENTS_DIR} for removal.")

        if relative_paths_to_remove:
            for doc_id in relative_paths_to_remove:
                # print(f"    - Queuing removal for ID: {doc_id}")
                remove_document_from_collection(doc_id)

    if files_to_process:
        # print("  - Processing and embedding new/modified files...")
        for file_path_str in files_to_process:
            relative_path = str(Path(file_path_str).relative_to(DOCUMENTS_DIR))
            print(f"    - Processing: {relative_path}")
            process_file_for_embeddings(file_path_str, DOCUMENTS_DIR)

    # print("Change processing complete.")
    save_current_state(current_state, STATE_FILE)
    return changes_made


if __name__ == "__main__":
    changes = check_files_state()
    if changes:
        print("\nFile check and processing finished: Embeddings updated.")
    else:
        print("\nFile check finished: No embedding updates needed.")

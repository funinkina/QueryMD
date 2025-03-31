import os
import json
from pathlib import Path
import toml
from embeddings_manager import remove_document_from_collection, process_file_for_embeddings

config = toml.load("config.toml")

TRACKING_METHOD = config['state_tracking'].get('method', 'mtime').lower()

if TRACKING_METHOD not in ["mtime", "git"]:
    raise ValueError(f"Invalid state_tracking method '{TRACKING_METHOD}' in config. Must be 'mtime' or 'git'.")

DOCUMENTS_DIR = Path(config["files"]["markdown_directory"]).resolve()
STATE_FILE = Path(config['files']['state_file']).resolve()


def load_previous_state_mtime(state_file_path):
    if state_file_path.exists():
        try:
            with open(state_file_path, 'r') as f:
                content = f.read()
                if not content:
                    return {}
                return json.loads(content)
        except json.JSONDecodeError:
            print(f"Warning: State file {state_file_path} is corrupted (mtime mode). Starting fresh.")
            return {}
        except Exception as e:
            print(f"Error loading state file {state_file_path} (mtime mode): {e}. Starting fresh.")
            return {}
    return {}


def save_current_state_mtime(state_data, state_file_path):
    try:
        state_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file_path, 'w') as f:
            json.dump(state_data, f, indent=4)
    except Exception as e:
        print(f"Error saving state file {state_file_path} (mtime mode): {e}")


def is_markdown_file_path(file_path_str):
    """Checks if a string path ends with a markdown extension."""
    extensions = ['.md', '.markdown']
    try:
        return Path(file_path_str).suffix.lower() in extensions
    except Exception:
        return False


def check_files_state_mtime():
    """Original state check based on mtime and size."""
    print(f"Using mtime tracking method for directory: {DOCUMENTS_DIR}")

    if not DOCUMENTS_DIR.is_dir():
        print(f"Error: Document directory not found or is not a directory: {DOCUMENTS_DIR}")
        return False

    abs_state_file = STATE_FILE.resolve()
    previous_state = load_previous_state_mtime(abs_state_file)
    current_state = {}
    found_files = set()
    files_to_process_abs = []
    files_to_remove_abs = []
    changes_made = False

    for root, _, filenames in os.walk(DOCUMENTS_DIR):
        if '.git' in Path(root).parts:
            continue

        for filename in filenames:
            file_path = Path(root) / filename
            if not is_markdown_file_path(str(file_path)):
                continue

            abs_path_str = str(file_path.resolve())
            found_files.add(abs_path_str)

            try:
                if not file_path.is_file():
                    continue

                current_mtime = os.path.getmtime(file_path)
                current_size = os.path.getsize(file_path)
                current_state[abs_path_str] = {
                    'mtime': current_mtime,
                    'size': current_size
                }

                prev_file_state = previous_state.get(abs_path_str)
                if not prev_file_state:
                    # print(f"Detected new file: {file_path.relative_to(DOCUMENTS_DIR)}") # Optional
                    files_to_process_abs.append(abs_path_str)
                elif (prev_file_state['mtime'] != current_mtime or prev_file_state['size'] != current_size):
                    # print(f"Detected modified file: {file_path.relative_to(DOCUMENTS_DIR)}") # Optional
                    files_to_process_abs.append(abs_path_str)

            except OSError as e:
                print(f"Error accessing file {file_path}: {e}. Skipping.")
            except Exception as e:
                print(f"Unexpected error processing file {file_path}: {e}. Skipping.")

    previous_files_abs = set(previous_state.keys())
    deleted_files_abs = previous_files_abs - found_files
    files_to_remove_abs.extend(list(deleted_files_abs))

    # if deleted_files_abs:
    #     deleted_relative = [str(Path(p).relative_to(DOCUMENTS_DIR)) for p in deleted_files_abs]
    #     print(f"Detected deleted files: {', '.join(deleted_relative)}")

    if not files_to_process_abs and not files_to_remove_abs:
        save_current_state_mtime(current_state, abs_state_file)
        return False

    print("\nProcessing detected changes (mtime)...")
    changes_made = True

    files_needing_removal_ids = set()

    for abs_path_str in files_to_remove_abs:
        if abs_path_str in previous_state:
            try:
                relative_path = str(Path(abs_path_str).relative_to(DOCUMENTS_DIR.resolve()))
                files_needing_removal_ids.add(relative_path)
            except ValueError:
                pass

    for abs_path_str in files_to_process_abs:
        if abs_path_str in previous_state:
            try:
                relative_path = str(Path(abs_path_str).relative_to(DOCUMENTS_DIR.resolve()))
                files_needing_removal_ids.add(relative_path)
            except ValueError:
                pass

    if files_needing_removal_ids:
        print(f"  - Removing embeddings for {len(files_needing_removal_ids)} file ID(s)...")
        for doc_id in files_needing_removal_ids:
            remove_document_from_collection(doc_id)

    if files_to_process_abs:
        print(f"  - Adding/updating embeddings for {len(files_to_process_abs)} file(s)...")
        for file_path_str in files_to_process_abs:
            process_file_for_embeddings(file_path_str, DOCUMENTS_DIR.resolve())

    print("Change processing complete (mtime).")
    save_current_state_mtime(current_state, abs_state_file)
    return changes_made

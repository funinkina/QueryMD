import json
from pathlib import Path
import toml
from embeddings_manager import remove_document_from_collection, process_file_for_embeddings
import git

config = toml.load("config.toml")

TRACKING_METHOD = config['state_tracking'].get('method', 'mtime').lower()

if TRACKING_METHOD not in ["mtime", "git"]:
    raise ValueError(f"Invalid state_tracking method '{TRACKING_METHOD}' in config. Must be 'mtime' or 'git'.")

DOCUMENTS_DIR = Path(config["files"]["markdown_directory"]).resolve()
STATE_FILE = Path(config['files']['state_file']).resolve()


def load_previous_state_git(state_file_path):
    """Loads the last processed Git commit SHA."""
    if state_file_path.exists():
        try:
            with open(state_file_path, 'r') as f:
                content = f.read()
                if not content:
                    return None
                state_data = json.loads(content)
                return state_data.get("last_processed_commit")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: State file {state_file_path} corrupted or invalid format ({e}). Re-processing all files.")
            return None
    return None


def save_current_state_git(commit_sha, state_file_path):
    """Saves the latest processed Git commit SHA."""
    try:
        state_file_path.parent.mkdir(parents=True, exist_ok=True)
        state_data = {"last_processed_commit": commit_sha}
        with open(state_file_path, 'w') as f:
            json.dump(state_data, f, indent=4)
    except Exception as e:
        print(f"Error saving state file {state_file_path}: {e}")


def is_markdown_file_path(file_path_str):
    """Checks if a string path ends with a markdown extension."""
    extensions = ['.md', '.markdown']
    try:
        return Path(file_path_str).suffix.lower() in extensions
    except Exception:
        return False


def check_files_state_git():
    """Checks file states based on Git commits and updates embeddings."""
    print(f"Using Git to track changes in: {DOCUMENTS_DIR}")
    changes_processed = False
    repo = None

    try:
        repo = git.Repo(DOCUMENTS_DIR)
        if repo.bare:
            print(f"Error: Directory {DOCUMENTS_DIR} is a bare Git repository. Cannot process files.")
            return False

    except git.InvalidGitRepositoryError:
        print(f"Error: Directory {DOCUMENTS_DIR} is not a valid Git repository.")
        print("Please ensure 'markdown_directory' in config.toml points to a Git repository root.")
        return False

    except Exception as e:
        print(f"Error initializing Git repository object at {DOCUMENTS_DIR}: {e}")
        return False

    try:
        current_head_sha = repo.head.commit.hexsha
    except Exception as e:
        print(f"Error getting current HEAD commit from repository: {e}")
        return False

    last_processed_sha = load_previous_state_git(STATE_FILE)
    print(f"  Current HEAD commit: {current_head_sha[:7]}")
    print(f"  Last processed commit: {last_processed_sha[:7] if last_processed_sha else 'None (processing all)'}")

    files_to_process = set()
    files_to_remove = set()

    if last_processed_sha == current_head_sha:
        print("No new commits detected. Embeddings are up-to-date.")
        return False

    try:
        commit_range = f"{last_processed_sha}..{current_head_sha}" if last_processed_sha else current_head_sha

        if last_processed_sha:
            try:
                last_commit_obj = repo.commit(last_processed_sha)
            except git.BadName:
                print(f"Warning: Last processed commit '{last_processed_sha}' not found in repository history. Re-processing all files.")
                last_processed_sha = None

        if last_processed_sha:
            print(f"Finding changes between {last_processed_sha[:7]} and {current_head_sha[:7]}...")
            diffs = repo.commit(last_processed_sha).diff(current_head_sha)

            for diff_item in diffs:
                if diff_item.change_type == 'D' and is_markdown_file_path(diff_item.a_path):
                    print(f"  - Detected deleted: {diff_item.a_path}")
                    files_to_remove.add(diff_item.a_path)

                elif diff_item.change_type in ('A', 'M', 'T') and is_markdown_file_path(diff_item.b_path):

                    if diff_item.change_type == 'M':
                        print(f"  - Detected modified: {diff_item.b_path}")
                        files_to_remove.add(diff_item.b_path)
                    elif diff_item.change_type == 'A':
                        print(f"  - Detected added: {diff_item.b_path}")
                    elif diff_item.change_type == 'T':
                        print(f"  - Detected type change to file: {diff_item.b_path}")
                    files_to_process.add(DOCUMENTS_DIR / diff_item.b_path)

                elif diff_item.change_type == 'R':
                    is_a_markdown = is_markdown_file_path(diff_item.a_path)
                    is_b_markdown = is_markdown_file_path(diff_item.b_path)
                    print(f"  - Detected renamed: {diff_item.a_path} -> {diff_item.b_path}")
                    if is_a_markdown:
                        files_to_remove.add(diff_item.a_path)  # Remove old path ID
                    if is_b_markdown:
                        files_to_process.add(DOCUMENTS_DIR / diff_item.b_path)  # Process new path

        else:
            print("Processing all tracked Markdown files...")
            tracked_files = repo.git.ls_files().splitlines()
            for file_rel_path in tracked_files:
                if is_markdown_file_path(file_rel_path):
                    files_to_process.add(DOCUMENTS_DIR / file_rel_path)

    except git.GitCommandError as e:
        print(f"Error executing Git command: {e}")
        return False
    except Exception as e:
        print(f"Error processing Git diff: {e}")
        return False

    if not files_to_process and not files_to_remove:
        print("No relevant Markdown file changes found in new commits.")
        if not last_processed_sha:
            save_current_state_git(current_head_sha, STATE_FILE)
        return False

    print("\nProcessing detected changes...")
    changes_processed = True

    if files_to_remove:
        print(f"  - Removing embeddings for {len(files_to_remove)} file(s)...")
        for relative_path_id in files_to_remove:
            remove_document_from_collection(relative_path_id)

    if files_to_process:
        print(f"  - Adding/updating embeddings for {len(files_to_process)} file(s)...")
        for abs_file_path in files_to_process:
            if abs_file_path.is_file():
                process_file_for_embeddings(str(abs_file_path), DOCUMENTS_DIR)
            else:
                print(f"    - Warning: File {abs_file_path} not found, skipping processing.")

    print(f"Updating last processed commit to: {current_head_sha[:7]}")
    save_current_state_git(current_head_sha, STATE_FILE)

    return changes_processed

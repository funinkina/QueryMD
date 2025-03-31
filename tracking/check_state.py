from pathlib import Path
import toml
from tracking.git_tracking import check_files_state_git
from tracking.mtime_tracking import check_files_state_mtime
import time

config = toml.load("config.toml")

TRACKING_METHOD = config['state_tracking'].get('method', 'mtime').lower()

if TRACKING_METHOD not in ["mtime", "git"]:
    print(f"Invalid state_tracking method '{TRACKING_METHOD}' in config. Must be 'mtime' or 'git'.")
    print("Defaulting to 'mtime' method.")
    TRACKING_METHOD = "mtime"

DOCUMENTS_DIR = Path(config["files"]["markdown_directory"]).resolve()
STATE_FILE = Path(config['files']['state_file']).resolve()


def check_files_state():
    """Checks file state using the method specified in config.toml"""
    if TRACKING_METHOD == "git":
        return check_files_state_git()
    elif TRACKING_METHOD == "mtime":
        return check_files_state_mtime()
    else:
        print(f"Error: Unknown tracking method '{TRACKING_METHOD}'. Defaulting to mtime.")
        return check_files_state_mtime()


if __name__ == "__main__":
    print(f"Running check_state standalone test (Method: {TRACKING_METHOD.upper()})")
    start_time = time.time()
    changes = check_files_state()
    end_time = time.time()
    if changes:
        print("\nFile check and processing finished: Embeddings updated.")
    else:
        print("\nFile check finished: No embedding updates needed.")
    print(f"Total time: {end_time - start_time:.2f} seconds.")

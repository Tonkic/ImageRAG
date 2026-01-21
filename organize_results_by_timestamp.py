import os
import shutil
import re
from datetime import datetime

RESULTS_DIR = "/home/tingyu/imageRAG/results"

def organize():
    # Only process directories in the root of results/
    for folder_name in os.listdir(RESULTS_DIR):
        folder_path = os.path.join(RESULTS_DIR, folder_name)

        # Skip files
        if not os.path.isdir(folder_path):
            continue

        # Check for logs/run_config.txt
        config_path = os.path.join(folder_path, "logs", "run_config.txt")
        if not os.path.exists(config_path):
            # Likely an already organized folder or a folder without config
            continue

        timestamp_str = None
        try:
            with open(config_path, 'r') as f:
                for line in f:
                    if line.strip().startswith("Timestamp:"):
                        timestamp_str = line.split("Timestamp:", 1)[1].strip()
                        break
        except Exception as e:
            print(f"Error reading {config_path}: {e}")
            continue

        if not timestamp_str:
            print(f"Skipping {folder_name}: No Timestamp found in config.")
            continue

        try:
            # Parse timestamp, e.g., "2026-01-06 03:58:33"
            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

            # Construct target directory name: YYYY.M.D (e.g., 2026.1.6)
            target_dir_name = f"{dt.year}.{dt.month}.{dt.day}"
            target_dir_path = os.path.join(RESULTS_DIR, target_dir_name)

            # Create target directory if it doesn't exist
            os.makedirs(target_dir_path, exist_ok=True)

            # Construct new folder name with time suffix: FolderName_HH-MM-SS
            time_suffix = dt.strftime("%H-%M-%S")
            new_folder_name = f"{folder_name}_{time_suffix}"
            new_folder_path = os.path.join(target_dir_path, new_folder_name)

            # Avoid overwriting or moving into itself
            if folder_path == new_folder_path:
                print(f"Skipping {folder_name}: Already in correct location.")
                continue

            if os.path.exists(new_folder_path):
                print(f"Warning: Target {new_folder_path} already exists. Skipping move for {folder_name}.")
                continue

            print(f"Moving '{folder_name}' -> '{target_dir_name}/{new_folder_name}'")
            shutil.move(folder_path, new_folder_path)

        except ValueError as e:
            print(f"Error parsing timestamp '{timestamp_str}' in {folder_name}: {e}")
        except Exception as e:
            print(f"Error moving {folder_name}: {e}")

if __name__ == "__main__":
    organize()

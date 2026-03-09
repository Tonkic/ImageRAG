import os
import glob
import shutil
import argparse
import datetime
import time

def get_experiment_details(path):
    """
    Returns (num_classes, last_modified_date) for a given experiment directory.
    num_classes is computed by finding unique prefixes among _FINAL.png and _V*.png.
    """
    if not os.path.isdir(path):
        return 0, None

    generated_classes = set()
    latest_time = 0

    # Process all image files indicating generation progress
    for p in glob.glob(os.path.join(path, "*_FINAL.png")):
        base = os.path.basename(p)
        class_name = base.replace("_FINAL.png", "")
        generated_classes.add(class_name)
        mtime = os.path.getmtime(p)
        if mtime > latest_time:
            latest_time = mtime

    for p in glob.glob(os.path.join(path, "*_V*.png")):
        base = os.path.basename(p)
        # Handle cases where multiple versions exist (e.g., _V0, _V1)
        try:
            class_name = base.rsplit("_V", 1)[0]
            generated_classes.add(class_name)
            mtime = os.path.getmtime(p)
            if mtime > latest_time:
                latest_time = mtime
        except Exception:
            pass

    if latest_time == 0:
        # Fallback to log/txt modification time if no generation images found
        # Check both the directory itself and a 'logs' subdirectory
        log_files = glob.glob(os.path.join(path, "*.log")) + \
                    glob.glob(os.path.join(path, "*.txt")) + \
                    glob.glob(os.path.join(path, "logs", "*.log")) + \
                    glob.glob(os.path.join(path, "logs", "*.txt"))
        for p in log_files:
            mtime = os.path.getmtime(p)
            if mtime > latest_time:
                latest_time = mtime

    if latest_time == 0:
        # Fallback to directory modification time if completely empty
        latest_time = os.path.getmtime(path)

    last_mod_date = datetime.datetime.fromtimestamp(latest_time).date()
    return len(generated_classes), last_mod_date, latest_time

def main():
    parser = argparse.ArgumentParser(description="Clean up incomplete experiments (Not from today and < 100 classes).")
    parser.add_argument("--root_dir", type=str, default="/home/tingyu/imageRAG/results/", help="Root directory containing experiment folders.")
    parser.add_argument("--delete", action="store_true", help="If specified, actually delete the folders. Otherwise, just print what would be deleted.")
    parser.add_argument("--include_today", action="store_true", help="If specified, also delete incomplete experiments modified today.")
    args = parser.parse_args()

    root_dir = args.root_dir
    if not os.path.exists(root_dir):
        print(f"Error: Directory {root_dir} does not exist.")
        return

    today = datetime.date.today()
    candidate_folders = []

    print(f"Scanning '{root_dir}' for experiment folders...")

    # Flexible heuristic to find experiment folders:
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Prevent accidentally processing a 'logs' dir if it somehow slipped through
        if os.path.basename(dirpath) == "logs":
            continue

        has_imgs = any(f.endswith("_FINAL.png") or ("_V" in f and f.endswith(".png")) for f in filenames)
        has_logs = any(f.endswith(".log") or f.endswith(".txt") for f in filenames) or "logs" in dirnames

        rel_path = os.path.relpath(dirpath, root_dir)
        depth = len(rel_path.split(os.sep)) if rel_path != "." else 0

        is_empty = len(dirnames) == 0 and len(filenames) == 0

        # We consider a dir an experiment folder if:
        # 1. It directly has generated images
        # 2. OR it's at depth >= 2 and contains logs or is completely empty
        is_exp = has_imgs or ((has_logs or is_empty) and depth >= 2)

        if is_exp:
            class_count, last_mod_date, latest_time = get_experiment_details(dirpath)
            candidate_folders.append({
                "path": dirpath,
                "count": class_count,
                "date": last_mod_date,
                "latest_time": latest_time
            })

            # Prune search: do not descend into subdirectories (e.g., 'logs') of an identified experiment folder
            dirnames.clear()

    if not candidate_folders:
        print("No experiment directories containing generated images were found.")
        return

    current_time = time.time()
    
    # Filter rules: classes < 100 AND 
    # (not strictly today OR include_today is True) AND 
    # (has NOT been modified in the last 4 hours (14400 seconds), protecting active runs)
    incomplete_old_folders = [
        exp for exp in candidate_folders
        if exp["count"] < 100 
        and (exp["date"] != today or args.include_today)
        and (current_time - exp["latest_time"] > 14400)
    ]

    print("\n" + "="*50)
    print("--- Summary of Found Experiments ---")
    print(f"Total experiment folders found: {len(candidate_folders)}")
    target_msg = "Target incomplete folders (<100 classes"
    target_msg += ", including today, but inactive > 4hr): " if args.include_today else ", not today, inactive > 4hr): "
    print(f"{target_msg}{len(incomplete_old_folders)}")
    print("="*50 + "\n")

    if len(incomplete_old_folders) > 0:
        for exp in incomplete_old_folders:
            folder_name = os.path.relpath(exp['path'], root_dir)
            print(f"- TARGET: {folder_name:<50} | Classes: {exp['count']:>3}/100 | Last Modified: {exp['date']}")

        if args.delete:
            print("\n🚨 DELETE MODE ACTIVE 🚨")
            confirmation = input(f"Are you sure you want to permanently delete these {len(incomplete_old_folders)} folder(s)? (y/n): ")
            if confirmation.lower() == 'y':
                print("\nDeleting incomplete experiments...")
                success_count = 0
                for exp in incomplete_old_folders:
                    folder_path = exp["path"]
                    folder_name = os.path.relpath(folder_path, root_dir)
                    try:
                        shutil.rmtree(folder_path)
                        print(f"  [Deleted] {folder_name}")
                        success_count += 1
                    except Exception as e:
                        print(f"  [Failed]  {folder_name}: {e}")
                print(f"\nCleanup finished. Successfully deleted {success_count} folders.")
            else:
                print("\nDeletion aborted by user.")
        else:
            print("\n[INFO] This is a DRY-RUN. No files were deleted.")
            print("----> To actually delete these directories, run the script with the '--delete' flag.")
    else:
        print("\nAll clear! No old, incomplete experiments found.")

if __name__ == "__main__":
    main()

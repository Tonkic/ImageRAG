import os
import argparse
import subprocess
import sys

# 服务器配置 (根据您之前的脚本推断)
HOST = "202.121.140.48"
PORT = "20563"
USERNAME = "tingyu"
REMOTE_PATH = "/home/tingyu/imageRAG"

# 在根目录明确需要删除的文件
ROOT_FILES_TO_DELETE = [
    "_patch_input_interpreter.py",
    "_find_dead_code.py",
    "src/retrieval/global_memory.py",
]

# 严格白名单：在 src/experiments/ 目录下，除了这些文件，其它所有 .py 都会被删除！
EXPERIMENTS_TO_KEEP = [
    "OmniGenV2_Ablation_noDINO.py",
    "OmniGenV2_Ablation_noDINO_Aircraft_AR.py",
    "OmniGenV2_Ablation_noInputDecomp.py",
    "OmniGenV2_Ablation_noInputDecomp_Aircraft_AR.py",
    "OmniGenV2_Ablation_noTAC.py",
    "OmniGenV2_Ablation_noTAC_Aircraft_AR.py",
    "OmniGenV2_Ablation_noVAR.py",
    "OmniGenV2_Ablation_noVAR_Aircraft_AR.py",
    "OmniGenV2_TAC_DINO_Importance_Aircraft.py",
    "OmniGenV2_TAC_DINO_Importance_Aircraft_AR.py",
    "ZImageDemo.py",
    "test_flux_kontext.py"
]

def run_cmd(cmd_string):
    """根据执行环境决定运行本地或通过 SSH"""
    # 如果当前就在服务器上运行，就直接在本地执行
    in_server = "tingyu" in os.getcwd() or "imageRAG" in os.getcwd()
    
    if in_server:
        result = subprocess.run(cmd_string, shell=True, capture_output=True, text=True)
    else:
        ssh_cmd = ["ssh", "-p", PORT, f"{USERNAME}@{HOST}", cmd_string]
        result = subprocess.run(ssh_cmd, capture_output=True, text=True)
    return result

def clean_remote(execute_delete=False):
    print("=" * 60)
    print("🧹 ImageRAG Server Cleanup Script")
    print("=" * 60)
    
    print(f"Targeting: {USERNAME}@{HOST}:{PORT}")
    print(f"Remote Path: {REMOTE_PATH}\n")

    # 准备目标列表
    target_files = list(ROOT_FILES_TO_DELETE)
    
    # 动态获取服务器上 src/experiments 下的违规文件
    print("Scanning remote src/experiments/ directory for un-whitelisted files...")
    exp_dir = f"{REMOTE_PATH}/src/experiments" if not ("tingyu" in os.getcwd() or "imageRAG" in os.getcwd()) else "src/experiments"
    ls_cmd = f"ls -1 {exp_dir}/*.py 2>/dev/null"
    res = run_cmd(ls_cmd)
    
    if res.stdout:
        remote_py_files = res.stdout.strip().split('\n')
        for py_path in remote_py_files:
            file_name = py_path.split('/')[-1]
            if file_name not in EXPERIMENTS_TO_KEEP:
                target_files.append(f"src/experiments/{file_name}")

    if not target_files:
        print("\nYour server is perfectly clean! No files need deletion. 🎉")
        return

    print(f"\nFound {len(target_files)} illicit files scheduled for cleanup:")
    for file in target_files:
        print(f"  - {file}")
    print("\n" + "=" * 60)
    
    if not execute_delete:
        print("\n[DRY RUN MODE] No files were actually deleted.")
        print("To execute the deletion, please run the script with the --delete flag:")
        print("    python sync_cleanup_server.py --delete")
        print("=" * 60)
        return

    print("\n[DELETE MODE] Starting deletion phase...")
    success_count = 0
    skip_count = 0
    
    for file in target_files:
        remote_file_path = f"{REMOTE_PATH}/{file}" if not ("tingyu" in os.getcwd() or "imageRAG" in os.getcwd()) else file
        # 确认文件是否存在的安全删除命令
        delete_cmd = f"if [ -f '{remote_file_path}' ]; then rm -f '{remote_file_path}' && echo 'DELETED'; else echo 'NOT_FOUND'; fi"
        
        res = run_cmd(delete_cmd)
        result_text = res.stdout.strip()
        
        if "DELETED" in result_text:
            print(f"  [√] Deleted: {file}")
            success_count += 1
        elif "NOT_FOUND" in result_text:
            print(f"  [-] Skipped (Not Found): {file}")
            skip_count += 1
        else:
            print(f"  [X] Error on {file}: {res.stderr.strip()}")

    # 删除残留的 .bak 文件
    print(f"\nCleaning up any stray .bak files in src/experiments/...")
    bak_path = f"{REMOTE_PATH}/src/experiments/*.bak" if not ("tingyu" in os.getcwd() or "imageRAG" in os.getcwd()) else "src/experiments/*.bak"
    bak_cmd = f"rm -f {bak_path}"
    run_cmd(bak_cmd)
    print("  [√] Removed .bak files.")

    print("=" * 60)
    print(f"Cleanup finished! Successfully deleted {success_count} files ({skip_count} were already gone).")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up old ImageRAG python scripts on the server.")
    parser.add_argument("--delete", action="store_true", help="Actually execute the deletion commands.")
    args = parser.parse_args()

    clean_remote(execute_delete=args.delete)


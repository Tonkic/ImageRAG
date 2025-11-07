import os

def generate_file_list_with_limit(root_dir='.', output_file='project_structure.txt'):
    """
    生成项目文件列表并保存到文件。
    当目录中的文件或子目录数超过阈值时，不一一列出，只显示总数。

    :param root_dir: 项目的根目录。
    :param output_file: 输出文件的名称。
    """

    # --- 配置项 ---
    # 1. 设置文件数量阈值。
    FILE_COUNT_THRESHOLD = 20
    # 2. 新增：设置子目录数量阈值。
    SUBDIR_COUNT_THRESHOLD = 15

    # 3. 常见的需要忽略的目录
    IGNORED_DIRS = {'.git', '.vscode', '__pycache__', 'node_modules', '.idea', 'venv', '.next'}
    # 4. 常见的需要忽略的文件
    IGNORED_FILES = {'.DS_Store'}
    # --- 配置结束 ---

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            project_name = os.path.basename(os.path.abspath(root_dir))
            f.write(f"项目 '{project_name}' 的文件结构：\n")
            f.write(f"(当子目录数 > {SUBDIR_COUNT_THRESHOLD} 或 文件数 > {FILE_COUNT_THRESHOLD} 时，仅显示摘要)\n")
            f.write("=" * 60 + "\n\n")

            # 使用 topdown=True 是关键，因为它允许我们在遍历过程中修改 sub_directories 列表
            for current_path, sub_directories, files_in_current_path in os.walk(root_dir, topdown=True):
                # --- 过滤需要忽略的目录 ---
                sub_directories[:] = sorted([d for d in sub_directories if d not in IGNORED_DIRS])

                # 如果当前目录本身就在忽略列表中，则跳过
                if os.path.basename(current_path) in IGNORED_DIRS:
                    continue

                relative_path = os.path.relpath(current_path, root_dir)
                depth = 0 if relative_path == '.' else len(relative_path.split(os.sep))

                indent = "    " * depth
                # 根目录特殊处理，显示为 './'
                dir_name = os.path.basename(current_path) if current_path != root_dir else './'
                f.write(f"{indent}📂 {dir_name}/\n")

                file_indent = "    " * (depth + 1)

                # --- 核心逻辑 1：检查子目录数量 ---
                num_subdirs = len(sub_directories)
                if 0 < num_subdirs > SUBDIR_COUNT_THRESHOLD:
                    # 如果子目录数量大于阈值，打印摘要并阻止 os.walk 继续深入
                    f.write(f"{file_indent}📚 包含 {num_subdirs} 个子目录 (数量过多，不一一列出)\n")
                    # 清空列表，这样 os.walk 就不会访问这些子目录了
                    sub_directories[:] = []

                # --- 核心逻辑 2：检查文件数量 ---
                printable_files = sorted([fn for fn in files_in_current_path if fn not in IGNORED_FILES])
                num_files = len(printable_files)

                if 0 < num_files > FILE_COUNT_THRESHOLD:
                    # 如果文件数量大于阈值，只打印摘要信息
                    f.write(f"{file_indent}📦 包含 {num_files} 个文件 (数量过多，不一一列出)\n")
                else:
                    # 否则，逐一列出文件
                    for filename in printable_files:
                        f.write(f"{file_indent}📄 {filename}\n")

        print(f"✅ 文件列表已成功导出到: {output_file}")

    except IOError as e:
        print(f"❌ 错误：无法写入文件 {output_file}。原因: {e}")

if __name__ == "__main__":
    # 确保在项目根目录运行此脚本
    project_path = '.'
    generate_file_list_with_limit(project_path)
import scipy.io
import pandas as pd
import numpy as np
import os

def load_mat_and_convert(mat_path):
    if not os.path.exists(mat_path):
        print(f"Error: File not found at {mat_path}")
        return

    print(f"Loading {mat_path}...")
    try:
        mat_data = scipy.io.loadmat(mat_path)
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return

    # 排除元数据 key
    keys = [k for k in mat_data.keys() if not k.startswith('__')]
    print(f"Found variables: {keys}")

    # 在当前目录下创建输出文件夹
    output_dir = "mat_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}")

    for key in keys:
        data = mat_data[key]
        print(f"\nProcessing variable: '{key}' | Type: {type(data)} | Shape: {getattr(data, 'shape', 'N/A')}")

        # 尝试处理 ImageNet meta.mat 中常见的 'synsets' 结构化数组
        # 或者其他类似的 numpy structured array
        if isinstance(data, np.ndarray) and (data.dtype.names or key == 'synsets'):
            try:
                # 获取数组内容，通常是 (N, 1) 或 (1, N)
                vals = data.ravel()

                parsed_data = []
                # 获取字段名
                dtype_names = vals.dtype.names if vals.dtype.names else []

                if dtype_names:
                    print(f"Detected structured array with fields: {dtype_names}")
                    for item in vals:
                        entry = {}
                        for name in dtype_names:
                            val = item[name]

                            # 处理常见的嵌套由 mat 导致的数据格式问题
                            if isinstance(val, np.ndarray):
                                if val.size == 0:
                                    entry[name] = None
                                elif val.size == 1:
                                    # 尝试提取单个值
                                    if val.dtype.kind in {'U', 'S'}: # String
                                        entry[name] = str(val[0])
                                    else:
                                        entry[name] = val.item()
                                else:
                                    # 数组转字符串
                                    entry[name] = str(val.tolist())
                            else:
                                entry[name] = val
                        parsed_data.append(entry)

                    df = pd.DataFrame(parsed_data)
                    csv_path = os.path.join(output_dir, f"{key}.csv")
                    df.to_csv(csv_path, index=False)
                    print(f"--> Saved table to {csv_path}")
                    continue
            except Exception as e:
                print(f"Warning: Failed to parse structured data for '{key}': {e}. Falling back to default dump.")

        # 通用数组处理
        if isinstance(data, np.ndarray):
            # 如果是简单的 2D 数组，直接存 CSV
            if data.ndim <= 2 and not data.dtype.names:
                try:
                    df = pd.DataFrame(data)
                    csv_path = os.path.join(output_dir, f"{key}.csv")
                    df.to_csv(csv_path, index=False, header=False)
                    print(f"--> Saved array to {csv_path}")
                except Exception as e:
                    print(f"Warning: Failed to save array to CSV: {e}")
            else:
                # 维度高或者无法直接转换的，保存为 TXT 文本描述
                txt_path = os.path.join(output_dir, f"{key}.txt")
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"Shape: {data.shape}\n")
                    f.write(f"Dtype: {data.dtype}\n")
                    f.write("Content:\n")
                    f.write(str(data))
                print(f"--> Saved complex array content to {txt_path}")
        else:
             # 非数组数据，直接转字符串保存
            txt_path = os.path.join(output_dir, f"{key}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(str(data))
            print(f"--> Saved content to {txt_path}")

if __name__ == "__main__":
    target_file = '/home/tingyu/imageRAG/datasets/ILSVRC2012_train/ILSVRC2012_devkit_t12/data/meta.mat'
    load_mat_and_convert(target_file)

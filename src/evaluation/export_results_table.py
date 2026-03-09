
import os
import glob
import pandas as pd
import argparse

def parse_metrics(file_path):
    metrics = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        try:
                            val = float(parts[1].strip())
                            metrics[key] = val
                        except: pass
    except: pass
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/home/tingyu/imageRAG/results")
    parser.add_argument("--output_file", type=str, default="consolidated_results.csv")
    args = parser.parse_args()

    # Find all evaluation_metrics.txt
    # Pattern: root_dir/*/*/*/*/logs/evaluation_metrics.txt (Depth varies)
    # Better to walk

    records = []

    print(f"Scanning {args.root_dir}...")
    for root, dirs, files in os.walk(args.root_dir):
        if "evaluation_metrics.txt" in files:
            log_path = os.path.join(root, "evaluation_metrics.txt")

            # Experiment Name: Rel path from results
            # e.g. LongCLIP/2026.2.6/OmniGenV2_TAC_VAR_Aircraft_12-38-24/logs/evaluation_metrics.txt
            rel_path = os.path.relpath(log_path, args.root_dir)
            # Remove /logs/evaluation_metrics.txt
            exp_name = os.path.dirname(os.path.dirname(rel_path))

            # Parse
            met = parse_metrics(log_path)
            if met:
                met['Experiment'] = exp_name
                records.append(met)

    if not records:
        print("No metrics found.")
        return

    df = pd.DataFrame(records)

    # Reorder columns usually preferred
    cols = ['Experiment'] + [c for c in df.columns if c != 'Experiment']
    df = df[cols]

    # Save CSV
    df.to_csv(args.output_file, index=False)
    print(f"Saved CSV to {args.output_file}")

    # Save Markdown
    md_file = args.output_file.replace('.csv', '.md')
    df.to_markdown(md_file, index=False)
    print(f"Saved Markdown to {md_file}")

    # Print
    print("\n" + df.to_markdown(index=False))

if __name__ == "__main__":
    main()

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EVALSCOPE_METRICS = [
    "CLIPScore", "ImageReward", "PickScore",
    "VQAScore", "BLIPv2Score", "HPSv2Score",
    "HPSv2.1Score", "MPS", "FGA_BLIP2Score",
]

SCORE_PATTERNS = [
    re.compile(r"\[IPC\]\s+score=([0-9]+(?:\.[0-9]+)?)"),
    re.compile(r"\[Step4-TAC\].*?score=([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"\[Step4-TAC\]\s+Score=([0-9]+(?:\.[0-9]+)?)"),
]
SAVE_PATTERN = re.compile(r"Saved\s+(.+_V(\d+)\.png)")


def safe_name_to_class_name(safe_name: str) -> str:
    return safe_name.replace("-", "/") if "/" in safe_name else safe_name.replace("_", " ")


def ensure_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def parse_best_scored_image(log_path: Path) -> tuple[Path | None, dict[int, float]]:
    exp_dir = log_path.parent
    current_version = None
    version_scores: dict[int, float] = {}

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            save_match = SAVE_PATTERN.search(line)
            if save_match:
                current_version = int(save_match.group(2))
                continue

            if current_version is None:
                continue

            for pattern in SCORE_PATTERNS:
                m = pattern.search(line)
                if m:
                    version_scores.setdefault(current_version, float(m.group(1)))
                    break

    if not version_scores:
        return None, {}

    best_version = max(version_scores.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    best_image = exp_dir / f"{log_path.stem}_V{best_version}.png"
    if not best_image.exists():
        return None, version_scores
    return best_image, version_scores


def collect_selection_images(exp_dir: Path) -> tuple[dict[str, Path], dict[str, Path], dict[str, float]]:
    final_map: dict[str, Path] = {}
    best_map: dict[str, Path] = {}
    best_scores: dict[str, float] = {}

    for log_path in sorted(exp_dir.glob("*.log")):
        safe_name = log_path.stem
        final_path = exp_dir / f"{safe_name}_FINAL.png"
        if final_path.exists():
            final_map[safe_name] = final_path

        best_path, version_scores = parse_best_scored_image(log_path)
        if best_path is not None:
            best_map[safe_name] = best_path
            best_version = int(best_path.stem.rsplit("_V", 1)[1])
            best_scores[safe_name] = version_scores[best_version]
        elif final_path.exists():
            best_map[safe_name] = final_path

    return final_map, best_map, best_scores


def build_selection_dir(target_dir: Path, selected: dict[str, Path]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "logs").mkdir(exist_ok=True)
    for safe_name, src_path in selected.items():
        dst_path = target_dir / f"{safe_name}_FINAL.png"
        ensure_link_or_copy(src_path, dst_path)


def run_metrics(selection_root: Path, args) -> None:
    from src.evaluation.evaluate_evalscope import (
        run_single_metric,
        run_fid_is_metric,
        run_dino_metric,
    )

    metric_names = [m.strip() for m in args.metrics.split(",")]
    if "all" in metric_names:
        metric_names = list(EVALSCOPE_METRICS)

    for metric_name in metric_names:
        if metric_name in {"FID Score", "Inception Score", "DINO v3 Score"}:
            continue
        run_single_metric(
            metric_name,
            str(selection_root),
            args.classes_txt,
            args.device_id,
            force_rerun=True,
            allow_incomplete=True,
        )

    if "all" in args.metrics or "DINO v3 Score" in args.metrics:
        run_dino_metric(
            str(selection_root),
            args.classes_txt,
            args.real_images_list,
            args.real_images_root,
            args.dinov3_repo_path,
            args.dinov3_weights_path,
            args.device_id,
            force_rerun=True,
            allow_incomplete=True,
        )

    if "all" in args.metrics or "FID Score" in args.metrics or "Inception Score" in args.metrics:
        run_fid_is_metric(
            str(selection_root),
            args.classes_txt,
            args.real_images_list,
            args.real_images_root,
            args.device_id,
            force_rerun=True,
            allow_incomplete=True,
        )


def load_summary(exp_path: Path) -> dict:
    metrics_path = exp_path / "logs" / "evalscope_metrics.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("summary", {})


def print_summary(label: str, summary: dict) -> None:
    print(f"\n=== {label} ===")
    for key in [
        "BLIPv2Score", "CLIPScore", "DINO v3 Score", "FGA_BLIP2Score",
        "FID Score", "HPSv2.1Score", "HPSv2Score", "ImageReward",
        "Inception Score", "MPS", "PickScore", "VQAScore",
    ]:
        if key in summary:
            print(f"{key}: {summary[key]:.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare experiment FINAL images vs highest-log-score images.")
    parser.add_argument("--exp_dir", required=True, help="Experiment directory, e.g. results/.../OmniGenV2_IPC_AR_04-47-44")
    parser.add_argument("--device_id", default="0")
    parser.add_argument("--metrics", default="all", help="Comma-separated metric names or 'all'")
    parser.add_argument("--classes_txt", default="datasets/fgvc-aircraft-2013b/data/variants.txt")
    parser.add_argument("--real_images_list", default="datasets/fgvc-aircraft-2013b/data/images_variant_test.txt")
    parser.add_argument("--real_images_root", default="datasets/fgvc-aircraft-2013b/data/images")
    parser.add_argument("--dinov3_repo_path", default="/home/tingyu/imageRAG/dinov3")
    parser.add_argument("--dinov3_weights_path", default="/home/tingyu/imageRAG/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    parser.add_argument("--work_dir", default=None, help="Optional persistent work directory. Default uses temp/")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    if not exp_dir.is_dir():
        raise SystemExit(f"Experiment directory not found: {exp_dir}")

    final_map, best_map, best_scores = collect_selection_images(exp_dir)
    if not final_map:
        raise SystemExit(f"No *_FINAL.png files found in {exp_dir}")
    if not best_map:
        raise SystemExit(f"No scored images could be parsed from logs in {exp_dir}")

    base_work = Path(args.work_dir) if args.work_dir else PROJECT_ROOT / "temp"
    base_work.mkdir(parents=True, exist_ok=True)
    compare_root = Path(tempfile.mkdtemp(prefix=f"compare_{exp_dir.name}_", dir=str(base_work)))
    final_dir = compare_root / "final_selection"
    best_dir = compare_root / "bestscore_selection"

    build_selection_dir(final_dir, final_map)
    build_selection_dir(best_dir, best_map)

    print(f"[INFO] Working directory: {compare_root}")
    print(f"[INFO] FINAL images: {len(final_map)}")
    print(f"[INFO] BEST-SCORE images: {len(best_map)}")

    examples = list(best_scores.items())[:10]
    if examples:
        print("\n[INFO] Example best-score picks from logs:")
        for safe_name, score in examples:
            print(f"  {safe_name}: score={score:.2f} -> {best_map[safe_name].name}")

    run_metrics(compare_root, args)

    final_summary = load_summary(final_dir)
    best_summary = load_summary(best_dir)

    print_summary("FINAL selection", final_summary)
    print_summary("BEST-SCORE selection", best_summary)

    print("\n=== delta (best_score - final) ===")
    keys = sorted(set(final_summary.keys()) | set(best_summary.keys()))
    for key in keys:
        if key in final_summary and key in best_summary:
            print(f"{key}: {best_summary[key] - final_summary[key]:+.6f}")

    print(f"\n[INFO] Compare artifacts kept at: {compare_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

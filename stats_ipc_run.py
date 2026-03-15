import argparse
import json
import re
from collections import Counter
from pathlib import Path


STEP5_RE = re.compile(r">>> STEP 5: Retry \(max=(\d+), TIFA=(ON|OFF)\)")
IPC_RE = re.compile(r"\[IPC\] score=([0-9.]+), same_id=(True|False), accepted=(True|False)")
RETRY_ACCEPT_RE = re.compile(r"\[Step5\] ACCEPT retry=(\d+)")
FORCED_ACCEPT_RE = re.compile(r"\[Step5\] retry=(\d+) ground truth ACCEPT")
FINAL_RE = re.compile(r">>> FINAL: (.+) score=([0-9.\-]+)")


def parse_log(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    ipc_matches = IPC_RE.findall(text)
    ipc_records = [
        {
            "score": float(score),
            "same_id": same_id == "True",
            "accepted": accepted == "True",
        }
        for score, same_id, accepted in ipc_matches
    ]

    step5_match = STEP5_RE.search(text)
    retry_accept_match = RETRY_ACCEPT_RE.search(text)
    forced_accept_match = FORCED_ACCEPT_RE.search(text)
    final_match = FINAL_RE.search(text)

    status = "incomplete"
    accepted_retry = None
    forced_retry = None

    if retry_accept_match:
        status = "passed_on_retry"
        accepted_retry = int(retry_accept_match.group(1))
    elif forced_accept_match:
        status = "forced_accept"
        forced_retry = int(forced_accept_match.group(1))
    elif ipc_records and ipc_records[0]["accepted"]:
        status = "passed_initial"
    elif final_match:
        status = "final_without_accept_marker"

    summary = {
        "class_name": log_path.stem,
        "status": status,
        "ipc_calls": len(ipc_records),
        "initial_ipc_score": ipc_records[0]["score"] if ipc_records else None,
        "initial_same_id": ipc_records[0]["same_id"] if ipc_records else None,
        "initial_accepted": ipc_records[0]["accepted"] if ipc_records else None,
        "last_ipc_score": ipc_records[-1]["score"] if ipc_records else None,
        "last_same_id": ipc_records[-1]["same_id"] if ipc_records else None,
        "last_accepted": ipc_records[-1]["accepted"] if ipc_records else None,
        "max_retries": int(step5_match.group(1)) if step5_match else 0,
        "tifa": step5_match.group(2) if step5_match else None,
        "accepted_retry": accepted_retry,
        "forced_retry": forced_retry,
        "final_image": final_match.group(1) if final_match else None,
        "final_score": float(final_match.group(2)) if final_match else None,
        "has_step5": step5_match is not None,
        "log_path": str(log_path),
    }

    summary_json = log_path.parent / "logs" / f"{log_path.stem}_summary.json"
    if summary_json.exists():
        try:
            summary["summary_json"] = json.loads(summary_json.read_text(encoding="utf-8"))
        except Exception:
            summary["summary_json"] = None
    else:
        summary["summary_json"] = None

    return summary


def print_bucket(title: str, rows: list[dict]) -> None:
    if not rows:
        return
    print(f"\n[{title}] {len(rows)}")
    for row in rows:
        extra = []
        if row["accepted_retry"] is not None:
            extra.append(f"retry={row['accepted_retry']}")
        if row["forced_retry"] is not None:
            extra.append(f"forced_retry={row['forced_retry']}")
        if row["initial_ipc_score"] is not None:
            extra.append(f"init_score={row['initial_ipc_score']:.2f}")
        if row["last_ipc_score"] is not None:
            extra.append(f"last_score={row['last_ipc_score']:.2f}")
        print(f"- {row['class_name']}" + (f" ({', '.join(extra)})" if extra else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize IPC slim run logs.")
    parser.add_argument("run_dir", type=str, help="Result directory containing per-class .log files.")
    parser.add_argument("--json", action="store_true", help="Print full JSON summary.")
    parser.add_argument("--show-classes", action="store_true", help="Print class lists for each bucket.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    logs = sorted(
        p for p in run_dir.glob("*.log")
        if p.is_file()
    )
    if not logs:
        raise SystemExit(f"No .log files found in: {run_dir}")

    rows = [parse_log(p) for p in logs]
    counts = Counter(row["status"] for row in rows)
    total = len(rows)
    true_pass = counts["passed_initial"] + counts["passed_on_retry"]
    forced = counts["forced_accept"]
    incomplete = counts["incomplete"] + counts["final_without_accept_marker"]

    overview = {
        "run_dir": str(run_dir),
        "total_classes": total,
        "passed_initial": counts["passed_initial"],
        "passed_on_retry": counts["passed_on_retry"],
        "true_pass_total": true_pass,
        "forced_accept": forced,
        "incomplete_or_unknown": incomplete,
        "true_pass_rate": round(true_pass / total, 4),
        "forced_accept_rate": round(forced / total, 4),
    }

    if args.json:
        print(json.dumps({"overview": overview, "rows": rows}, indent=2, ensure_ascii=False))
        return 0

    print("=== IPC Run Summary ===")
    for k, v in overview.items():
        print(f"{k}: {v}")

    if args.show_classes:
        print_bucket("passed_initial", [r for r in rows if r["status"] == "passed_initial"])
        print_bucket("passed_on_retry", [r for r in rows if r["status"] == "passed_on_retry"])
        print_bucket("forced_accept", [r for r in rows if r["status"] == "forced_accept"])
        print_bucket("incomplete_or_unknown", [r for r in rows if r["status"] in {"incomplete", "final_without_accept_marker"}])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

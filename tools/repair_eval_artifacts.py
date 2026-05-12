import argparse
import json
import os
import shutil


EVAL_SUFFIX_PATHS = {
    "summary.json": "summary_eval.json",
    "result.json": "result_eval.json",
    "analysis.json": "analysis_eval.json",
    "cases.json": "cases_eval.json",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def load_history(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--best_checkpoint", type=int, required=True)
    parser.add_argument("--status", default="completed")
    parser.add_argument("--train_time_sec", type=float, default=None)
    parser.add_argument("--avg_step_time_sec", type=float, default=None)
    parser.add_argument("--peak_gpu_memory_mb", type=float, default=None)
    args = parser.parse_args()

    summary_path = os.path.join(args.run_dir, "summary.json")
    history_path = os.path.join(args.run_dir, "train_history.jsonl")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(summary_path)
    if not os.path.exists(history_path):
        raise FileNotFoundError(history_path)

    current_summary = load_json(summary_path)

    for source_name, target_name in EVAL_SUFFIX_PATHS.items():
        source_path = os.path.join(args.run_dir, source_name)
        target_path = os.path.join(args.run_dir, target_name)
        if os.path.exists(source_path) and not os.path.exists(target_path):
            shutil.copy2(source_path, target_path)

    history = load_history(history_path)
    best_record = None
    for record in history:
        if record.get("step") == args.best_checkpoint:
            best_record = record
            break
    if best_record is None:
        best_records = [record for record in history if record.get("is_best")]
        best_record = best_records[-1] if best_records else None

    training_stats = current_summary.get("training_stats", {})
    training_stats.update({
        "status": args.status,
        "oom": False,
        "interrupted": False,
    })
    if args.train_time_sec is not None:
        training_stats["total_training_time_sec"] = args.train_time_sec
    if args.avg_step_time_sec is not None:
        training_stats["avg_step_time_sec"] = args.avg_step_time_sec
    if args.peak_gpu_memory_mb is not None:
        training_stats["peak_gpu_memory_mb"] = args.peak_gpu_memory_mb

    restored_summary = {
        "run_id": current_summary.get("run_id"),
        "eval_mode": False,
        "status": args.status,
        "best_checkpoint": args.best_checkpoint,
        "selected_checkpoint": args.best_checkpoint,
        "best_valid_metrics": best_record.get("metrics") if best_record else None,
        "test_metrics": None,
        "threshold_calibration": None,
        "training_stats": training_stats,
        "paths": current_summary.get("paths", {}),
        "repair_note": (
            "Original summary.json had been overwritten by eval mode. "
            "Eval artifacts were copied to *_eval.json; training summary was restored from train_history.jsonl."
        ),
    }
    write_json(restored_summary, summary_path)
    print(f"Repaired {summary_path}")


if __name__ == "__main__":
    main()

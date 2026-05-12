import datetime
import json
import os
import platform
import random
import re
import socket
import subprocess
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from transformers import EvalPrediction


def seed_everything(seed):
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_log(log, path):
    with open(path, 'a') as f:
        f.writelines(log + '\n')


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def write_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def append_jsonl(data, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=True) + "\n")


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.integer, )):
        return int(value)
    if isinstance(value, (np.floating, )):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "__dict__"):
        return {str(k): make_json_safe(v) for k, v in vars(value).items()}
    return str(value)


def get_git_commit(workdir):
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def package_version(pkg_name):
    try:
        module = __import__(pkg_name)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return "not-installed"


def slugify_run_part(value, default="run", max_length=48):
    value = str(value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    if not value:
        value = default
    return value[:max_length].strip("-") or default


def build_run_id(args):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment = getattr(args, "experiment_name", "") or getattr(args, "theory_variant", "")
    experiment = slugify_run_part(experiment, default="run")
    return f"{args.model}-{args.dataset}-{experiment}-seed{args.seed}-{timestamp}"


def collect_runtime_context(args, device, dataset_sizes, workdir):
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    return {
        "run_id": build_run_id(args),
        "timestamp": datetime.datetime.now().isoformat(),
        "command": " ".join(sys.argv),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "device": str(device),
        "gpu_name": gpu_name,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "transformers_version": package_version("transformers"),
        "pyg_version": package_version("torch_geometric"),
        "git_commit": get_git_commit(workdir),
        "dataset_sizes": dataset_sizes,
        "args": make_json_safe(vars(args)),
    }


def multi_class_metrics(predictions, labels):
    softmax = torch.nn.Softmax()
    probs = softmax(torch.Tensor(predictions))
    y_pred = np.argmax(probs, axis=-1)
    y_true = labels
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=None)
    w_f1 = f1_score(y_true, y_pred, average='weighted')
    ma_f1 = f1_score(y_true, y_pred, average="macro")
    mi_f1 = f1_score(y_true, y_pred, average="micro")
    # return as dictionary
    metrics = {'accuracy': accuracy, 'weighted-f1': w_f1, 'macro-f1': ma_f1, 'micro-f1': mi_f1}
    labels_set = set(np.unique(labels).tolist())
    for _id in labels_set:
        metrics[str(_id)] = f1[_id]
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_class_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result

"""Evaluate a model on the benchmarks listed in an experiment's config.yaml.

Usage:
    python -m slmpt.eval.run_eval --config experiments/00-baseline/config.yaml
    python -m slmpt.eval.run_eval --config experiments/00-baseline/config.yaml --limit 10  # smoke test
"""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from slmpt.eval.benchmarks import headline, run_task, wrap_model
from slmpt.models.load import get_device, load_model_and_tokenizer
from slmpt.utils.env import load_env
from slmpt.utils.seed import set_seed


def git_commit() -> str:
    """Short commit hash, with "-dirty" if there are uncommitted changes."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None, help="examples per task (overrides config)")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    eval_cfg = cfg["eval"]
    seed = cfg.get("seed", 42)
    limit = args.limit if args.limit is not None else eval_cfg.get("limit")

    # Smoke tests write to results/smoke/ so they never overwrite a real run.
    out_dir = args.config.parent / "results"
    if limit:
        out_dir = out_dir / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    load_env()
    set_seed(seed)

    model, tokenizer = load_model_and_tokenizer(**cfg["model"])
    lm = wrap_model(model, tokenizer, batch_size=eval_cfg.get("batch_size", 8))

    metrics, timings = {}, {}
    for task in eval_cfg["tasks"]:
        name = task["name"]
        print(f"\n=== {name} ===")
        start = time.time()
        output = run_task(
            lm,
            name=name,
            num_fewshot=task.get("num_fewshot"),
            limit=limit,
            apply_chat_template=eval_cfg.get("apply_chat_template", False),
            seed=seed,
        )
        timings[name] = round(time.time() - start, 1)
        metrics[name] = headline(output, name)

        # Keep every prompt/response: reading real outputs is how you spot problems.
        with open(out_dir / f"samples_{name}.jsonl", "w") as f:
            for subtask, samples in output["samples"].items():
                for sample in samples:
                    f.write(json.dumps({"task": subtask, **sample}, default=str) + "\n")

    report = {
        "metrics": metrics,
        "meta": {
            "experiment": args.config.parent.name,
            "model": cfg["model"]["name_or_path"],
            "device": get_device(),
            "limit": limit,
            "seed": seed,
            "git_commit": git_commit(),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seconds_per_task": timings,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== summary ===")
    for name, values in metrics.items():
        for key, value in values.items():
            if "_stderr" not in key:
                print(f"{name:15s} {key:32s} {value:.4f}")
    print(f"\nwrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

"""Thin wrapper around lm-evaluation-harness (lm-eval)."""

from typing import Any

import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import PreTrainedModel, PreTrainedTokenizerBase

# lm-eval reports many metrics per task; these are the ones we track on the leaderboard.
# Keys are "<metric>,<filter>" exactly as lm-eval names them.
HEADLINE_METRICS: dict[str, list[str]] = {
    # strict-match: answer must follow the "#### <number>" format; flexible: last number in output.
    "gsm8k": ["exact_match,strict-match", "exact_match,flexible-extract"],
    "ifeval": ["prompt_level_strict_acc,none", "inst_level_strict_acc,none"],
    "arc_challenge": ["acc_norm,none"],
    "mmlu": ["acc,none"],
}


def wrap_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, batch_size: int) -> HFLM:
    """Wrap an already-loaded model so lm-eval uses it (instead of loading its own copy)."""
    return HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)


def run_task(
    lm: HFLM,
    name: str,
    num_fewshot: int | None,
    limit: int | None,
    apply_chat_template: bool,
    seed: int,
) -> dict[str, Any]:
    """Run one benchmark and return lm-eval's raw output (results + per-sample logs)."""
    return lm_eval.simple_evaluate(
        model=lm,
        tasks=[name],
        num_fewshot=num_fewshot,
        limit=limit,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=apply_chat_template,
        log_samples=True,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
    )


def headline(output: dict[str, Any], name: str) -> dict[str, float]:
    """Pick the tracked metrics (and their standard errors) out of lm-eval's output."""
    results = output["results"][name]
    picked = {}
    for key in HEADLINE_METRICS.get(name, []):
        metric, filt = key.split(",")
        picked[key] = results[key]
        stderr = results.get(f"{metric}_stderr,{filt}")
        if isinstance(stderr, float):
            picked[f"{metric}_stderr,{filt}"] = stderr
    return picked

"""Load a causal language model and its tokenizer from a Hub id or local path."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def get_device() -> str:
    """Pick the best available device: CUDA GPU, then Apple GPU (MPS), then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    """Map a config string to a torch dtype. "auto" = bfloat16 on GPU/MPS, float32 on CPU."""
    if dtype != "auto":
        return getattr(torch, dtype)
    return torch.float32 if device == "cpu" else torch.bfloat16


def load_model_and_tokenizer(
    name_or_path: str,
    dtype: str = "auto",
    device: str | None = None,
    revision: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model + tokenizer, move the model to `device`, and put it in eval mode."""
    device = device or get_device()

    tokenizer = AutoTokenizer.from_pretrained(name_or_path, revision=revision)
    # Batched generation needs a pad token; many base models don't define one.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        name_or_path, revision=revision, dtype=resolve_dtype(dtype, device)
    ).to(device)
    model.eval()
    return model, tokenizer

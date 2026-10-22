"""Load secrets (HF_TOKEN, WANDB_API_KEY, ...) from the project's .env file."""

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_env() -> None:
    """Copy values from slm-post-training/.env into environment variables (existing ones win)."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)

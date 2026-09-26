"""Keep model caches local to this project; never require a Hugging Face token."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure_runtime():
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

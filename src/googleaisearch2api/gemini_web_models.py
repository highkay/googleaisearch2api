"""Gemini web model registry (mode ids from the frontend MODE_CATEGORY enum)."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from loguru import logger

DEFAULT_MODEL = "gemini-3.7-flash"


class ModelConfig(TypedDict):
    mode: int
    think: int
    desc: str
    extra: NotRequired[dict[int, object]]


# MODE_CATEGORY enum from the Gemini frontend JS source:
#   1=FAST, 2=THINKING, 3=PRO, 4=AUTO, 5=FAST_DYNAMIC_THINKING, 6=FLASH_LITE
MODELS: dict[str, ModelConfig] = {
    "gemini-3.7-flash": {
        "mode": 1,
        "think": 0,
        "desc": "Latest all-around model (Gemini 3.7 Flash)",
    },
    "gemini-3.6-flash": {
        "mode": 1,
        "think": 0,
        "desc": "All-around model (Gemini 3.6 Flash)",
    },
    "gemini-3.5-flash": {
        "mode": 1,
        "think": 0,
        "desc": "Alias for gemini-3.6-flash (backend upgraded)",
    },
    "gemini-3.5-flash-thinking": {
        "mode": 2,
        "think": 0,
        "desc": "Deep thinking mode, longest output (~20k chars)",
    },
    "gemini-3.1-pro": {
        "mode": 3,
        "think": 0,
        "desc": "Pro model (requires cookie for real routing)",
    },
    "gemini-3.1-pro-enhanced": {
        "mode": 3,
        "think": 0,
        "extra": {31: 2, 80: 3},
        "desc": "Pro with enhanced output (experimental)",
    },
    "gemini-auto": {
        "mode": 4,
        "think": 0,
        "desc": "Auto model selection",
    },
    "gemini-3.5-flash-thinking-lite": {
        "mode": 5,
        "think": 0,
        "desc": "Dynamic thinking with adaptive depth",
    },
    "gemini-flash-lite": {
        "mode": 6,
        "think": 0,
        "desc": "Lightweight fast model",
    },
}


def resolve_model(
    model_name: str, default: str = DEFAULT_MODEL
) -> tuple[str | None, int | None, int | None, str | None, dict[int, object] | None]:
    """Resolve a model name to (name, mode_id, think_mode, error, extra_fields).

    Unknown model names fall back to the default rather than erroring, since
    upstream clients may request arbitrary model identifiers.
    """
    think_override: int | None = None
    if "@think=" in model_name:
        model_name, think_str = model_name.rsplit("@think=", 1)
        try:
            think_override = int(think_str)
        except ValueError:
            return None, None, None, f"Invalid think level: {think_str}", None
    cfg = MODELS.get(model_name)
    if cfg is None:
        logger.warning("Unknown Gemini web model '{}', falling back to '{}'", model_name, default)
        model_name = default
        cfg = MODELS[default]
    think_mode = think_override if think_override is not None else cfg["think"]
    return model_name, cfg["mode"], think_mode, None, cfg.get("extra")

"""Ask Claude to translate menu items and judge them, returning strict JSON."""
from __future__ import annotations

import json
import logging
from typing import Literal

import anthropic
from pydantic import BaseModel

from .config import CLAUDE_MODEL

log = logging.getLogger(__name__)

FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-2", "claude-fable-5-1"}


class LlmItem(BaseModel):
    index: int
    name_en: str
    name_uz: str
    typical_ingredients: list[str]
    verdict: Literal["likely_ok", "doubtful", "haram"]
    reason_uz: str


class LlmResult(BaseModel):
    items: list[LlmItem]


SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "name_en": {"type": "string"},
                    "name_uz": {"type": "string"},
                    "typical_ingredients": {"type": "array", "items": {"type": "string"}},
                    "verdict": {"type": "string", "enum": ["likely_ok", "doubtful", "haram"]},
                    "reason_uz": {"type": "string"},
                },
                "required": ["index", "name_en", "name_uz", "typical_ingredients", "verdict", "reason_uz"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

SYSTEM = """You help Muslim visitors in Korea understand restaurant menus.
For each numbered menu item (Korean or English, possibly with OCR typos):
- name_en: short English name of the dish; name_uz: the same in Uzbek (Latin script).
- typical_ingredients: 3-6 ingredients this dish USUALLY contains in Korean restaurants.
- verdict:
  "haram"     - the dish normally contains pork, pork products, lard, or alcohol (incl. mirin/cooking wine in the sauce);
  "doubtful"  - contains beef, chicken, lamb or meat broth (slaughter method unknown), or commonly contains pork/alcohol depending on the restaurant;
  "likely_ok" - normally no meat and no alcohol (seafood and vegetable dishes).
- reason_uz: one short sentence in Uzbek explaining the verdict.
Be conservative: when unsure between two verdicts, pick the stricter one.
If an item is not food (e.g. a heading or opening hours), return verdict "likely_ok" and reason_uz "Taom emas".
Return one entry per input item, keeping its index."""


def analyze_items(names: list[str]) -> dict[int, LlmItem] | None:
    """Return {index: LlmItem}, or None if the call is not possible / fails."""
    if not names:
        return {}
    numbered = "\n".join(f"{i}. {n}" for i, n in enumerate(names))
    try:
        client = anthropic.Anthropic()
        # Server-side refusal fallback exists only on some models (not Haiku).
        extra = ({"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
                 if CLAUDE_MODEL in FALLBACK_MODELS else {})
        response = client.beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=16000,
            **extra,
            system=SYSTEM,
            messages=[{"role": "user", "content": f"Menu items:\n{numbered}"}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        )
    except TypeError as e:                 # raised by the SDK when no API key is configured
        if "authentication" not in str(e).lower():
            raise
        log.warning("ANTHROPIC_API_KEY is not set - running in rules-only mode")
        return None
    except anthropic.AuthenticationError:
        log.warning("Invalid Anthropic API key - running in rules-only mode")
        return None
    except anthropic.RateLimitError:
        log.warning("Claude rate limited - running in rules-only mode for this request")
        return None
    except anthropic.APIStatusError as e:
        log.error("Claude API error %s: %s", e.status_code, e.message)
        return None
    except anthropic.APIConnectionError:
        log.error("Cannot reach the Claude API")
        return None

    if response.stop_reason in ("refusal", "max_tokens"):
        log.warning("Claude stopped with %s", response.stop_reason)
        return None
    text = next((b.text for b in reversed(response.content) if b.type == "text"), None)
    if text is None:
        return None
    try:
        parsed = LlmResult.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValueError) as e:
        log.error("Bad JSON from Claude: %s", e)
        return None
    return {item.index: item for item in parsed.items if 0 <= item.index < len(names)}

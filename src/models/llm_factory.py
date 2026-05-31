from __future__ import annotations

import re
import time
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage

from .latency_optimizer import choose_profile
from .model_config import (
    DEFAULT_EXECUTION_MODE,
    DEFAULT_FAST_GROQ_MODEL,
    DEFAULT_GEMINI_EVAL_MODEL,
    DEFAULT_GEMINI_TEXT_MODEL,
    DEFAULT_GROQ_TEXT_MODEL,
)


# Output ceilings. Kept moderate so a single request stays well under Groq's
# free-tier 12k tokens/minute window (a ~650-word section is only ~900 tokens;
# 3072 is ample headroom without inflating the per-request rate-limit cost).
_SYNTHESIS_MAX_TOKENS = 3072
_EVAL_MAX_TOKENS = 1536

# Rate-limit handling. Groq free tier throttles on tokens-per-minute and returns
# a "try again in Ns" hint; Gemini returns a 'retryDelay'. We honor those hints
# (capped) and retry the same provider several times before falling through.
_RATE_LIMIT_HINTS = ("rate_limit", "rate limit", "resource_exhausted", "tokens per minute", " 429", "429 ")
_MAX_BACKOFF_SECONDS = 20.0


def _is_rate_limited(message: str) -> bool:
    low = message.lower()
    return any(hint in low for hint in _RATE_LIMIT_HINTS)


def _suggested_backoff(message: str, attempt: int) -> float:
    low = message.lower()
    for pattern in (r"try again in ([\d.]+)\s*s", r"please retry in ([\d.]+)s", r"retry in ([\d.]+)\s*s", r"retrydelay'?:?\s*'?([\d.]+)s"):
        match = re.search(pattern, low)
        if match:
            try:
                return min(float(match.group(1)) + 1.0, _MAX_BACKOFF_SECONDS)
            except ValueError:
                break
    return min(2.0 * (2 ** attempt), _MAX_BACKOFF_SECONDS)


def _groq_client(model: str, temperature: float, max_tokens: int):
    from langchain_groq import ChatGroq

    return ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens)


def _gemini_client(model: str, temperature: float, max_tokens: int, api_key: str):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
        max_output_tokens=max_tokens,
    )


def _build_clients(profile: str) -> list[Any]:
    import os

    google_api_key = os.getenv("GOOGLE_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    clients: list[Any] = []

    # Provider ordering note:
    # Groq is the PRIMARY text provider. Gemini's free tier is capped at ~20
    # requests/day, which the parallel section fan-out exhausts almost instantly
    # (see CLAUDE.md). Gemini is kept only as a secondary fallback so a single
    # run never collapses when one provider is rate-limited.

    if profile == "fast_eval":
        if groq_api_key:
            clients.append(_groq_client(DEFAULT_FAST_GROQ_MODEL, 0.2, _EVAL_MAX_TOKENS))
        if google_api_key:
            clients.append(_gemini_client(DEFAULT_GEMINI_EVAL_MODEL, 0.2, _EVAL_MAX_TOKENS, google_api_key))
        return clients

    if profile == "balanced_eval":
        # Use the small/fast model first for evaluation-style tasks (claim
        # extraction, fact-check, scoring, SEO). This conserves the scarce
        # 70B tokens-per-minute budget for actual writing/synthesis.
        if groq_api_key:
            clients.append(_groq_client(DEFAULT_FAST_GROQ_MODEL, 0.2, _EVAL_MAX_TOKENS))
        if google_api_key:
            clients.append(_gemini_client(DEFAULT_GEMINI_EVAL_MODEL, 0.2, _EVAL_MAX_TOKENS, google_api_key))
        if groq_api_key:
            clients.append(_groq_client(DEFAULT_GROQ_TEXT_MODEL, 0.2, _EVAL_MAX_TOKENS))
        return clients

    # quality_synthesis and balanced_synthesis (default): strongest writers first.
    if groq_api_key:
        clients.append(_groq_client(DEFAULT_GROQ_TEXT_MODEL, 0.4, _SYNTHESIS_MAX_TOKENS))
    if google_api_key:
        clients.append(_gemini_client(DEFAULT_GEMINI_TEXT_MODEL, 0.4, _SYNTHESIS_MAX_TOKENS, google_api_key))
    return clients


def get_clients(task_kind: str, execution_mode: str | None = None) -> list[Any]:
    mode = execution_mode or DEFAULT_EXECUTION_MODE
    profile = choose_profile(task_kind, mode)
    clients = _build_clients(profile)
    if not clients:
        raise RuntimeError("No supported LLM credentials found. Set GOOGLE_API_KEY or GROQ_API_KEY.")
    return clients


def response_to_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
        return "\n".join(chunk.strip() for chunk in chunks if chunk).strip()
    return str(content).strip()


def _attempt_loop(invoke_one, task_kind: str, execution_mode: str | None, attempts_per_client: int, what: str) -> tuple[Any, list[dict[str, Any]]]:
    """Try each client `attempts_per_client` times, honoring rate-limit backoff.

    `invoke_one(client)` performs a single call and returns its result. Raises
    RuntimeError with the last errors if every client/attempt is exhausted.
    """
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for client in get_clients(task_kind, execution_mode):
        provider = client.__class__.__name__
        for attempt in range(attempts_per_client):
            try:
                result = invoke_one(client)
                records.append({"provider": provider, "task_kind": task_kind, "attempt": attempt + 1, "succeeded": True})
                return result, records
            except Exception as exc:
                message = str(exc)
                errors.append(f"{provider} attempt {attempt + 1}: {message}")
                records.append({"provider": provider, "task_kind": task_kind, "attempt": attempt + 1, "succeeded": False, "error": message})
                # If throttled and we still have attempts left on this client,
                # wait out the suggested window and retry the SAME provider —
                # the alternate provider is often also exhausted.
                if attempt < attempts_per_client - 1 and _is_rate_limited(message):
                    time.sleep(_suggested_backoff(message, attempt))
    raise RuntimeError(f"All configured {what} models failed. " + " | ".join(errors[-4:]))


def invoke_text(
    messages: Sequence[BaseMessage],
    *,
    task_kind: str,
    execution_mode: str | None = None,
    attempts_per_client: int = 4,
) -> tuple[str, list[dict[str, Any]]]:
    return _attempt_loop(
        lambda client: response_to_text(client.invoke(list(messages))),
        task_kind,
        execution_mode,
        attempts_per_client,
        "text",
    )


def invoke_structured(
    schema: Any,
    messages: Sequence[BaseMessage],
    *,
    task_kind: str,
    execution_mode: str | None = None,
    attempts_per_client: int = 4,
) -> tuple[Any, list[dict[str, Any]]]:
    return _attempt_loop(
        lambda client: client.with_structured_output(schema).invoke(list(messages)),
        task_kind,
        execution_mode,
        attempts_per_client,
        "structured-output",
    )

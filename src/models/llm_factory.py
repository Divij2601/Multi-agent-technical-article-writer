from __future__ import annotations

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


def _build_clients(profile: str) -> list[Any]:
    import os

    google_api_key = os.getenv("GOOGLE_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    clients: list[Any] = []

    if profile == "fast_eval":
        if groq_api_key:
            from langchain_groq import ChatGroq

            clients.append(ChatGroq(model=DEFAULT_FAST_GROQ_MODEL, temperature=0.2))
        if google_api_key:
            from langchain_google_genai import ChatGoogleGenerativeAI

            clients.append(
                ChatGoogleGenerativeAI(
                    model=DEFAULT_GEMINI_EVAL_MODEL,
                    temperature=0.2,
                    google_api_key=google_api_key,
                )
            )
        return clients

    if profile == "balanced_eval":
        if google_api_key:
            from langchain_google_genai import ChatGoogleGenerativeAI

            clients.append(
                ChatGoogleGenerativeAI(
                    model=DEFAULT_GEMINI_EVAL_MODEL,
                    temperature=0.2,
                    google_api_key=google_api_key,
                )
            )
        if groq_api_key:
            from langchain_groq import ChatGroq

            clients.append(ChatGroq(model=DEFAULT_FAST_GROQ_MODEL, temperature=0.2))
        return clients

    if profile == "quality_synthesis":
        if google_api_key:
            from langchain_google_genai import ChatGoogleGenerativeAI

            clients.append(
                ChatGoogleGenerativeAI(
                    model=DEFAULT_GEMINI_TEXT_MODEL,
                    temperature=0.3,
                    google_api_key=google_api_key,
                )
            )
        if groq_api_key:
            from langchain_groq import ChatGroq

            clients.append(ChatGroq(model=DEFAULT_GROQ_TEXT_MODEL, temperature=0.3))
        return clients

    if groq_api_key:
        from langchain_groq import ChatGroq

        clients.append(ChatGroq(model=DEFAULT_GROQ_TEXT_MODEL, temperature=0.3))
    if google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        clients.append(
            ChatGoogleGenerativeAI(
                model=DEFAULT_GEMINI_TEXT_MODEL,
                temperature=0.3,
                google_api_key=google_api_key,
            )
        )
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


def invoke_text(
    messages: Sequence[BaseMessage],
    *,
    task_kind: str,
    execution_mode: str | None = None,
    attempts_per_client: int = 2,
) -> tuple[str, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for client in get_clients(task_kind, execution_mode):
        for attempt in range(attempts_per_client):
            try:
                text = response_to_text(client.invoke(list(messages)))
                records.append(
                    {
                        "provider": client.__class__.__name__,
                        "task_kind": task_kind,
                        "attempt": attempt + 1,
                        "succeeded": True,
                    }
                )
                return text, records
            except Exception as exc:
                errors.append(f"{client.__class__.__name__} attempt {attempt + 1}: {exc}")
                records.append(
                    {
                        "provider": client.__class__.__name__,
                        "task_kind": task_kind,
                        "attempt": attempt + 1,
                        "succeeded": False,
                        "error": str(exc),
                    }
                )
    raise RuntimeError("All configured text models failed. " + " | ".join(errors[-4:]))


def invoke_structured(
    schema: Any,
    messages: Sequence[BaseMessage],
    *,
    task_kind: str,
    execution_mode: str | None = None,
    attempts_per_client: int = 2,
) -> tuple[Any, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for client in get_clients(task_kind, execution_mode):
        runnable = client.with_structured_output(schema)
        for attempt in range(attempts_per_client):
            try:
                result = runnable.invoke(list(messages))
                records.append(
                    {
                        "provider": client.__class__.__name__,
                        "task_kind": task_kind,
                        "attempt": attempt + 1,
                        "succeeded": True,
                    }
                )
                return result, records
            except Exception as exc:
                errors.append(f"{client.__class__.__name__} attempt {attempt + 1}: {exc}")
                records.append(
                    {
                        "provider": client.__class__.__name__,
                        "task_kind": task_kind,
                        "attempt": attempt + 1,
                        "succeeded": False,
                        "error": str(exc),
                    }
                )
    raise RuntimeError("All configured structured-output models failed. " + " | ".join(errors[-4:]))

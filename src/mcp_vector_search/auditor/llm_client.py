"""Unified LLM client abstraction for the privacy auditor.

Supports both Anthropic (direct) and OpenRouter (OpenAI-compatible) backends.
The backend is selected via AuditorSettings.llm_backend.
"""

from __future__ import annotations

from loguru import logger

from .config import AuditorSettings


async def llm_complete(
    messages: list[dict[str, str]],
    model: str,
    settings: AuditorSettings,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> str:
    """Send a chat-completion request to the configured LLM backend.

    Args:
        messages: List of {"role": "system"|"user"|"assistant", "content": "..."} dicts.
        model: Model identifier (backend-specific format).
        settings: Auditor settings controlling which backend to use.
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.

    Returns:
        The text content of the model's response.

    Raises:
        ImportError: If required backend package is not installed.
        ValueError: If no API key is configured for the selected backend.
    """
    if settings.llm_backend == "openrouter":
        logger.debug("llm_complete: using openrouter backend, model=%s", model)
        return await _openrouter_complete(
            messages, model, settings, max_tokens, temperature
        )
    else:
        logger.debug("llm_complete: using anthropic backend, model=%s", model)
        return await _anthropic_complete(
            messages, model, settings, max_tokens, temperature
        )


async def _openrouter_complete(
    messages: list[dict[str, str]],
    model: str,
    settings: AuditorSettings,
    max_tokens: int,
    temperature: float,
) -> str:
    """Call OpenRouter's OpenAI-compatible chat completions API."""
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required for the OpenRouter backend. "
            "Install it with: pip install openai"
        ) from exc

    if not settings.openrouter_api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Export OPENROUTER_API_KEY or set MVS_AUDIT_OPENROUTER_API_KEY."
        )

    api_key = settings.openrouter_api_key.get_secret_value()
    client = AsyncOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=api_key,
    )

    response = await client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


async def _anthropic_complete(
    messages: list[dict[str, str]],
    model: str,
    settings: AuditorSettings,
    max_tokens: int,
    temperature: float,
) -> str:
    """Call Anthropic API directly using the anthropic SDK."""
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' package is required for the Anthropic backend. "
            "Install it with: pip install 'mcp-vector-search[auditor]'"
        ) from exc

    api_key = (
        settings.anthropic_api_key.get_secret_value()
        if settings.anthropic_api_key
        else ""
    )
    if not api_key:
        raise ValueError(
            "MVS_AUDIT_ANTHROPIC_API_KEY is not set. "
            "Export ANTHROPIC_API_KEY or set MVS_AUDIT_ANTHROPIC_API_KEY."
        )

    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Split system message out from the conversation
    system_msg = ""
    api_messages: list[dict[str, str]] = []
    for msg in messages:
        if msg["role"] == "system":
            system_msg = msg["content"]
        else:
            api_messages.append(msg)

    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_msg if system_msg else anthropic.NOT_GIVEN,
        messages=api_messages,  # type: ignore[arg-type]
    )
    return response.content[0].text

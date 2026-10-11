"""LLM providers behind one streaming interface, with a keyless extractive fallback.

Provider choice: DOCUMIND_LLM_PROVIDER if set, otherwise the first provider with a key
(Anthropic → Google Gemini → Groq → OpenAI → Azure OpenAI). With no key the app runs in
**extractive** mode: answers are assembled from the best-matching source sentences, so retrieval,
reranking and citations are all real and the demo never breaks on quota.
Gemini, Groq and Ollama are reached through their OpenAI-compatible endpoints."""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app.config import Settings, get_settings

logger = logging.getLogger("documind.llm")

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "google": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.2",
}
BASE_URLS = {
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1",
}


def provider_errors() -> tuple[type[BaseException], ...]:
    """Errors that mean "this provider failed, fall back". The SDKs are imported lazily: a keyless
    deployment never loads them, which saves ~45 MB of memory."""
    import anthropic
    import openai

    return (anthropic.APIError, openai.APIError, RuntimeError, TimeoutError)


@dataclass(frozen=True)
class Provider:
    name: str
    model: str
    api_key: str | None = None
    base_url: str | None = None

    @property
    def live(self) -> bool:
        return self.name != "extractive"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    extra: dict = field(default_factory=dict)


EXTRACTIVE = Provider("extractive", "extractive")


def resolve_provider(settings: Settings | None = None) -> Provider:
    s = settings or get_settings()
    keys = {
        "anthropic": s.anthropic_api_key,
        "google": s.google_api_key,
        "groq": s.groq_api_key,
        "openai": s.openai_api_key,
        "azure": s.azure_openai_api_key if s.azure_openai_endpoint and s.azure_openai_deployment else None,
    }
    name = s.llm_provider or next((n for n, k in keys.items() if k), "extractive")
    if name == "extractive":
        return EXTRACTIVE
    if name == "azure":
        return Provider(
            "azure",
            s.llm_model or s.azure_openai_deployment or "",
            s.azure_openai_api_key,
            s.azure_openai_endpoint,
        )
    model = s.llm_model or DEFAULT_MODELS.get(name, "")
    if name == "ollama":
        return Provider("ollama", model, api_key="ollama", base_url=s.ollama_url)
    return Provider(name, model, api_key=keys.get(name), base_url=BASE_URLS.get(name))


def _openai_client(p: Provider, timeout: float):
    import openai

    if p.name == "azure":
        return openai.AsyncAzureOpenAI(
            api_key=p.api_key,
            azure_endpoint=p.base_url,
            timeout=timeout,
            api_version=get_settings().azure_openai_api_version,
            max_retries=1,
        )
    return openai.AsyncOpenAI(api_key=p.api_key, base_url=p.base_url, timeout=timeout, max_retries=1)


async def stream_completion(
    p: Provider, system: str, user: str, max_tokens: int, usage: Usage | None = None
) -> AsyncIterator[str]:
    """Yields text deltas. Fills `usage` when the stream ends."""
    usage = usage if usage is not None else Usage()
    timeout = get_settings().llm_timeout_seconds
    if p.name == "anthropic":
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=p.api_key, timeout=timeout, max_retries=1)
        async with client.messages.stream(
            model=p.model,
            max_tokens=max_tokens,
            # The instructions never change between requests, so cache them.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "low"},  # grounded Q&A over given sources: no deep reasoning
        ) as stream:
            async for text in stream.text_stream:
                yield text
            final = await stream.get_final_message()
        if final.stop_reason == "refusal":
            raise RuntimeError("Model declined the request")
        usage.input_tokens, usage.output_tokens = final.usage.input_tokens, final.usage.output_tokens
        return

    client = _openai_client(p, timeout)
    stream = await client.chat.completions.create(
        model=p.model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )
    async for event in stream:
        if event.usage:
            usage.input_tokens = event.usage.prompt_tokens
            usage.output_tokens = event.usage.completion_tokens
        if event.choices and event.choices[0].delta.content:
            yield event.choices[0].delta.content


async def complete(p: Provider, system: str, user: str, max_tokens: int = 400) -> str:
    return "".join([t async for t in stream_completion(p, system, user, max_tokens)])

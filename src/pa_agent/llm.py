"""Provider-agnostic chat model factory (OpenAI or Anthropic via LangChain).

LLM_PROVIDER=none (the default) runs the deterministic rule-based path, so the
project works and its tests pass without an API key.
"""
from __future__ import annotations

from .config import Settings


def get_chat_model(s: Settings):
    if s.llm_provider in ("", "none"):
        return None
    if s.llm_provider == "openai":
        from langchain_openai import ChatOpenAI  # pip install langchain-openai
        return ChatOpenAI(model=s.default_model(), temperature=0, timeout=30, max_retries=2)
    if s.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # pip install langchain-anthropic
        return ChatAnthropic(model=s.default_model(), temperature=0, timeout=30, max_retries=2)
    raise ValueError(f"Unsupported LLM_PROVIDER={s.llm_provider!r} (use none, openai or anthropic)")

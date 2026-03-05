from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()

Provider = Literal["openai", "anthropic"]


def _get_env_var(name: str) -> str:
    """Fetch a required environment variable or raise a clear error."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_llm(provider: Provider, model_name: str, temperature: float = 0.0) -> BaseChatModel:
    """
    Factory for constructing a LangChain-compatible chat LLM.

    This decouples the scoring engine from any specific provider so that
    NSF reliability studies can compare scoring consistency across models
    by swapping the `provider` and `model_name` parameters.

    Parameters
    ----------
    provider:
        The LLM provider identifier. Supported values: ``\"openai\"``, ``\"anthropic\"``.
    model_name:
        The concrete model to use (e.g., ``\"gpt-4.1\"``, ``\"gpt-4o\"``, ``\"claude-3-opus-20240229\"``).
    temperature:
        Sampling temperature for the underlying chat model (defaults to 0.0 for deterministic scoring).

    Returns
    -------
    BaseChatModel
        A LangChain chat model instance (`ChatOpenAI` or `ChatAnthropic`) implementing the standard interface.
    """
    provider_normalized = provider.lower()

    if provider_normalized == "openai":
        api_key = _get_env_var("OPENAI_API_KEY")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )

    if provider_normalized == "anthropic":
        api_key = _get_env_var("ANTHROPIC_API_KEY")
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )

    raise ValueError(f"Unsupported provider '{provider}'. Expected 'openai' or 'anthropic'.")


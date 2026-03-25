from dataclasses import dataclass
import os
from typing import Any, Optional, Tuple

from google import genai
import openai

from shinka.env import load_shinka_dotenv

from .providers.pricing import get_provider

load_shinka_dotenv()

TIMEOUT = 600
_OPENROUTER_PREFIX = "openrouter/"


@dataclass(frozen=True)
class ResolvedEmbeddingModel:
    original_model_name: str
    api_model_name: str
    provider: str
    base_url: Optional[str] = None


def resolve_embedding_backend(model_name: str) -> ResolvedEmbeddingModel:
    """Resolve runtime backend info for embedding model identifiers."""
    if model_name.startswith("local/"):
        # Format: local/model@base_url
        parts = model_name.split("/", 1)[1].split("@", 1)
        api_model_name = parts[0]
        base_url = parts[1] if len(parts) > 1 else None
        return ResolvedEmbeddingModel(
            original_model_name=model_name,
            api_model_name=api_model_name,
            provider="local",
            base_url=base_url,
        )

    if model_name.startswith("gpu/"):
        # Format: gpu/all-MiniLM-L6-v2
        api_model_name = model_name.split("/", 1)[1]
        return ResolvedEmbeddingModel(
            original_model_name=model_name,
            api_model_name=api_model_name,
            provider="gpu",
        )

    provider = get_provider(model_name)
    if provider == "azure":
        api_model_name = model_name.split("azure-", 1)[-1]
        return ResolvedEmbeddingModel(
            original_model_name=model_name,
            api_model_name=api_model_name,
            provider=provider,
        )
    if provider is not None:
        return ResolvedEmbeddingModel(
            original_model_name=model_name,
            api_model_name=model_name,
            provider=provider,
        )
    if model_name.startswith(_OPENROUTER_PREFIX):
        api_model_name = model_name.split(_OPENROUTER_PREFIX, 1)[-1]
        if not api_model_name:
            raise ValueError(
                "OpenRouter embedding model is missing after 'openrouter/'."
            )
        return ResolvedEmbeddingModel(
            original_model_name=model_name,
            api_model_name=api_model_name,
            provider="openrouter",
        )
    raise ValueError(f"Embedding model {model_name} not supported.")


def get_client_embed(model_name: str) -> Tuple[Any, str]:
    """Get the client and model for the given embedding model name."""
    resolved = resolve_embedding_backend(model_name)
    provider = resolved.provider

    if provider == "gpu":
        from .local_gpu import GPUEmbeddingClient
        client = GPUEmbeddingClient(model_name=resolved.api_model_name)
    elif provider == "openai":
        client = openai.OpenAI(timeout=TIMEOUT)
    elif provider == "local":
        api_key = os.getenv("LOCAL_OPENAI_API_KEY", "local")
        base_url = os.getenv("SHINKA_EMBEDDING_BASE_URL", resolved.base_url)
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=TIMEOUT,
        )
    elif provider == "azure":
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_ENDPOINT"),
            timeout=TIMEOUT,
        )
    elif provider == "google":
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    elif provider == "openrouter":
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            timeout=TIMEOUT,
        )
    else:
        raise ValueError(f"Embedding model {model_name} not supported.")

    return client, resolved.api_model_name


def get_async_client_embed(model_name: str) -> Tuple[Any, str]:
    """Get the async client and model for the given embedding model name."""
    resolved = resolve_embedding_backend(model_name)
    provider = resolved.provider

    if provider == "gpu":
        from .local_gpu import GPUEmbeddingClient
        client = GPUEmbeddingClient(model_name=resolved.api_model_name)
    elif provider == "openai":
        client = openai.AsyncOpenAI()
    elif provider == "local":
        api_key = os.getenv("LOCAL_OPENAI_API_KEY", "local")
        base_url = os.getenv("SHINKA_EMBEDDING_BASE_URL", resolved.base_url)
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=TIMEOUT,
        )
    elif provider == "azure":
        client = openai.AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_ENDPOINT"),
        )
    elif provider == "google":
        # Gemini doesn't have async client yet, will use thread pool in embedding.py
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    elif provider == "openrouter":
        client = openai.AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            timeout=TIMEOUT,
        )
    else:
        raise ValueError(f"Embedding model {model_name} not supported.")

    return client, resolved.api_model_name

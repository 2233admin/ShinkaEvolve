import logging

logger = logging.getLogger(__name__)


def count_tokens(text: str, model: str = "") -> int:
    """Count tokens with tiktoken fallback to chars/4."""
    if any(m in model.lower() for m in ["gpt", "text-embedding-3", "o1", "o3"]):
        try:
            import tiktoken
            try:
                enc = tiktoken.encoding_for_model(model)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            pass
    return len(text) // 4


def get_model_context_limit(model: str) -> int:
    """Return context window for known models, default 128k."""
    limits = {
        "gpt-5-mini": 1048576,
        "gpt-5.4": 1048576,
        "gpt-4.1": 1047576,
        "gpt-4.1-mini": 1047576,
        "gemini-3-flash-preview": 1048576,
        "gemini-3.1-pro-preview": 2097152,
        "gemini-2.5-flash": 1048576,
        "gemini-2.5-pro": 1048576,
        "claude-sonnet-4-20250514": 200000,
        "claude-opus-4-20250514": 200000,
        "deepseek-chat": 65536,
        "deepseek-reasoner": 65536,
    }
    for m, limit in limits.items():
        if m in model:
            return limit
    if model.startswith("local/"):
        return 32768
    return 128000

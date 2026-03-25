from dataclasses import dataclass
import logging
from .token_counter import count_tokens, get_model_context_limit

logger = logging.getLogger(__name__)


@dataclass
class GuardResult:
    truncated_meta: str
    truncated_inspirations: str
    was_truncated: bool
    token_count: int


class ContextGuard:
    def __init__(self, model: str, max_ratio: float = 0.6):
        self.model = model
        self.limit = int(get_model_context_limit(model) * max_ratio)

    def _truncate_by_line(self, text: str, target_tokens: int) -> str:
        lines = text.splitlines()
        while lines and count_tokens("\n".join(lines), self.model) > target_tokens:
            lines.pop()
        return "\n".join(lines)

    def guard(
        self,
        system_prompt: str,
        user_prompt: str,
        meta_recommendations: str = "",
        inspirations: str = "",
    ) -> GuardResult:
        base_tokens = count_tokens(system_prompt + user_prompt, self.model)
        if base_tokens > self.limit:
            logger.warning(
                "Base prompt (%d tokens) already exceeds limit (%d)!",
                base_tokens, self.limit,
            )

        current_meta = meta_recommendations
        current_insp = inspirations

        total = base_tokens + count_tokens(current_meta + current_insp, self.model)
        was_truncated = False

        if total > self.limit:
            was_truncated = True
            # Truncate meta first
            meta_limit = max(0, self.limit - base_tokens - count_tokens(current_insp, self.model))
            current_meta = self._truncate_by_line(current_meta, meta_limit)

            # If still over, truncate inspirations
            total = base_tokens + count_tokens(current_meta + current_insp, self.model)
            if total > self.limit:
                insp_limit = max(0, self.limit - base_tokens - count_tokens(current_meta, self.model))
                current_insp = self._truncate_by_line(current_insp, insp_limit)

        final_count = base_tokens + count_tokens(current_meta + current_insp, self.model)
        return GuardResult(current_meta, current_insp, was_truncated, final_count)

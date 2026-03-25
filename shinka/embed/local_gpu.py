import logging
from typing import List, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class GPUEmbeddingClient:
    """GPU-accelerated embedding using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cuda"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local GPU embeddings. "
                "Install it with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.device = device
        logger.info(f"Loading local GPU embedding model '{model_name}' on {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = 64

    def get_embedding(
        self, code: Union[str, List[str]]
    ) -> Tuple[Union[List[float], List[List[float]]], float]:
        if isinstance(code, str):
            texts = [code]
            is_single = True
        else:
            texts = code
            is_single = False

        embeddings = self.encode_batch(texts, batch_size=self.batch_size)

        if is_single:
            return embeddings[0].tolist(), 0.0
        else:
            return embeddings.tolist(), 0.0

    def encode_batch(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

import logging
import sqlite3
import json
from typing import List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class FaissIndex:
    """Fast similarity search using faiss (IndexFlatIP + L2 norm = cosine sim)."""

    def __init__(self, dim: int):
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu or faiss-gpu is required for FaissIndex. "
                "Install it with: pip install faiss-cpu"
            )

        self.dim = dim
        self.quantizer = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIDMap2(self.quantizer)

    def add(self, program_id: int, embedding: List[float]):
        import faiss
        if not embedding or len(embedding) != self.dim:
            return

        vec = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vec)

        ids = np.array([program_id], dtype=np.int64)
        self.index.add_with_ids(vec, ids)

    def search(self, query: List[float], k: int = 10) -> List[Tuple[int, float]]:
        import faiss
        if self.index.ntotal == 0:
            return []

        vec = np.array([query], dtype=np.float32)
        faiss.normalize_L2(vec)

        scores, ids = self.index.search(vec, k)

        results = []
        for score, pid in zip(scores[0], ids[0]):
            if pid != -1:
                results.append((int(pid), float(score)))
        return results

    def compute_similarities(self, query: List[float]) -> List[float]:
        import faiss
        if self.index.ntotal == 0:
            return []

        vec = np.array([query], dtype=np.float32)
        faiss.normalize_L2(vec)

        k = self.index.ntotal
        scores, _ = self.index.search(vec, k)
        return scores[0].tolist()

    def remove(self, program_id: int):
        ids_to_remove = np.array([program_id], dtype=np.int64)
        self.index.remove_ids(ids_to_remove)

    def save(self, path: str):
        import faiss
        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path: str, dim: int) -> "FaissIndex":
        import faiss
        instance = cls(dim)
        instance.index = faiss.read_index(path)
        return instance

    def __len__(self) -> int:
        return self.index.ntotal


def build_index_from_db(db_path: str, island_idx: int) -> Optional[FaissIndex]:
    """Build a Faiss index from all programs in a specific island that have embeddings."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, embedding FROM programs WHERE island_idx = ? AND embedding IS NOT NULL",
            (island_idx,),
        )
        rows = cursor.fetchall()

        if not rows:
            conn.close()
            return None

        first_emb = json.loads(rows[0][1])
        dim = len(first_emb)

        faiss_idx = FaissIndex(dim)

        for pid, emb_json in rows:
            emb = json.loads(emb_json)
            faiss_idx.add(pid, emb)

        conn.close()
        logger.info(f"Built Faiss index with {len(faiss_idx)} vectors for island {island_idx}")
        return faiss_idx

    except Exception as e:
        logger.error(f"Failed to build Faiss index from DB: {e}")
        return None

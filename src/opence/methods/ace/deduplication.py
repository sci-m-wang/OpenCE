"""Semantic deduplication for playbook bullets."""

from __future__ import annotations

from typing import Dict, List

try:  # Optional heavy dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
except ImportError:  # pragma: no cover - fallback logic used
    SentenceTransformer = None  # type: ignore[assignment]
    cosine_similarity = None  # type: ignore[assignment]

class Deduplicator:
    """Finds semantically similar bullets using embeddings."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.8,
    ) -> None:
        self._model = SentenceTransformer(model_name) if SentenceTransformer else None
        self._threshold = similarity_threshold

    def find_duplicates(
        self,
        new_bullets: Dict[str, str],
        existing_bullets: Dict[str, str],
    ) -> List[str]:
        """
        Identifies new bullets that are semantically similar to existing ones.

        Args:
            new_bullets: A dictionary of {bullet_id: content} for new bullets.
            existing_bullets: A dictionary of {bullet_id: content} for existing bullets.

        Returns:
            A list of bullet IDs from `new_bullets` that are duplicates.
        """
        if not new_bullets or not existing_bullets:
            return []

        new_contents = list(new_bullets.values())
        existing_contents = list(existing_bullets.values())
        new_ids = list(new_bullets.keys())

        if self._model and cosine_similarity:
            new_embeddings = self._model.encode(new_contents, convert_to_tensor=True)
            existing_embeddings = self._model.encode(existing_contents, convert_to_tensor=True)
            similarities = cosine_similarity(new_embeddings, existing_embeddings)
            duplicate_ids = []
            for i, similarity_row in enumerate(similarities):
                if any(similarity > self._threshold for similarity in similarity_row):
                    duplicate_ids.append(new_ids[i])
            return duplicate_ids

        # Fallback: simple exact-match/substring heuristic (keeps tests lightweight)
        duplicates: List[str] = []
        existing_lower = [text.lower() for text in existing_contents]
        for idx, content in zip(new_ids, new_contents):
            normalized = content.lower()
            if any(normalized == other or normalized in other or other in normalized for other in existing_lower):
                duplicates.append(idx)
        return duplicates

"""Embeddings interchangeables pour le recommandeur de vins.

 ``sbert``  – embeddings **contextuels** (Sentence-BERT multilingue). Backend recommandé : c'est
               celui qui répond pleinement à l'exigence « embeddings contextuels (BERT, SBERT) ».

Il renvoie des vecteurs normalisés L2 : le produit scalaire est donc la similarité cosinus.
"""
from __future__ import annotations

import os
import unicodedata
from abc import ABC, abstractmethod
from typing import Iterable, Sequence

import numpy as np

DEFAULT_SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

FRENCH_STOPWORDS = [
    "a", "à", "au", "aux", "avec", "ce", "ces", "cet", "cette", "dans", "de", "des", "du", "elle", "en",
    "et", "est", "il", "ils", "je", "j", "la", "le", "les", "leur", "lui", "ma", "mais", "me", "même",
    "mes", "moi", "mon", "ne", "nos", "notre", "nous", "on", "ou", "par", "pas", "pour", "qu", "que",
    "qui", "sa", "se", "ses", "son", "sur", "ta", "te", "tes", "toi", "ton", "tu", "un", "une", "vos",
    "votre", "vous", "d", "l", "s", "c", "n", "m", "t", "y", "ça", "plus", "très", "peu", "aussi",
    "comme", "sans", "sous", "entre", "vers", "chez", "si", "ainsi", "afin", "être", "avoir", "fait",
    "faire", "veux", "voudrais", "cherche", "souhaite", "aime", "aimerais", "bien", "bon", "vin", "vins",
]


def _strip_accents(w: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", w) if not unicodedata.combining(ch))


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Normalise chaque ligne ; un vecteur nul reste nul (similarité 0, jamais NaN)."""
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class EmbeddingBackend(ABC):
    """Interface commune : ``fit`` (optionnel) puis ``encode``."""

    name: str = "base"
    label: str = "base"
    kind: str = ""          # famille de technique (pour l'affichage / le rapport)
    description: str = ""

    def fit(self, corpus: Sequence[str]) -> "EmbeddingBackend":  # noqa: D401 - no-op par défaut
        return self

    @abstractmethod
    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """Retourne une matrice (n_textes, dim) de vecteurs normalisés L2."""


# --------------------------------------------------------------------------------------- SBERT
class SBERTBackend(EmbeddingBackend):
    name = "sbert"
    kind = "Embeddings contextuels (SBERT)"

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer  # import tardif : dépendance lourde

        self.model_name = model_name or os.environ.get("WINE_SBERT_MODEL", DEFAULT_SBERT_MODEL)
        self.model = SentenceTransformer(self.model_name)
        # Les profils sensoriels dépassent parfois 128 tokens : on élargit la fenêtre.
        self.model.max_seq_length = max(getattr(self.model, "max_seq_length", 128) or 128, 256)
        self.label = f"SBERT · {self.model_name.split('/')[-1]}"
        self.description = "Embeddings de phrases contextuels, multilingues (français inclus)."

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts = [t if t.strip() else " " for t in texts]
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return l2_normalize(vecs)


# --------------------------------------------------------------------------------------- factory

def load_backend(name: str = "auto") -> tuple[EmbeddingBackend, list[str]]:
    """
    Retourne ``(SBERT, notes)`` où ``notes`` liste les replis effectués (à afficher à l'utilisateur).
    """
    notes: list[str] = []
    return SBERTBackend()

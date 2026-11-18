"""Backends d'embeddings interchangeables pour le recommandeur de vins.

Trois niveaux, du plus riche au plus léger (tous exposent la même interface) :

* ``sbert``  – embeddings **contextuels** (Sentence-BERT multilingue). Backend recommandé : c'est
               celui qui répond pleinement à l'exigence « embeddings contextuels (BERT, SBERT) ».
* ``spacy``  – embeddings **de mots** (vecteurs statiques fastText/word2vec de ``fr_core_news_md``),
               moyennés par texte. Sert de point de comparaison « Word2Vec/fastText ».
* ``lsa``    – TF-IDF + SVD tronquée (Latent Semantic Analysis), 100 % hors-ligne, uniquement
               scikit-learn. Repli de secours : sémantique latente apprise sur le corpus du projet.

Chaque backend renvoie des vecteurs normalisés L2 : le produit scalaire est donc la similarité cosinus.
"""
from __future__ import annotations

import os
import unicodedata
from abc import ABC, abstractmethod
from typing import Iterable, Sequence

import numpy as np

DEFAULT_SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_SPACY_MODEL = "fr_core_news_md"

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


# --------------------------------------------------------------------------------------- spaCy
class SpacyBackend(EmbeddingBackend):
    name = "spacy"
    kind = "Embeddings de mots (fastText/word2vec moyennés)"

    def __init__(self, model_name: str | None = None):
        import spacy

        self.model_name = model_name or os.environ.get("WINE_SPACY_MODEL", DEFAULT_SPACY_MODEL)
        self.nlp = spacy.load(self.model_name, disable=["parser", "ner"])
        if self.nlp.vocab.vectors.shape[0] == 0:
            raise RuntimeError(f"Le modèle spaCy {self.model_name} ne contient pas de vecteurs de mots.")
        self.dim = self.nlp.vocab.vectors.shape[1]
        self.label = f"spaCy · {self.model_name} (vecteurs de mots)"
        self.description = "Moyenne des vecteurs de mots (sans mots vides) – ne tient pas compte du contexte."

    def _vec(self, doc) -> np.ndarray:
        vs = [t.vector for t in doc if t.is_alpha and not t.is_stop and t.has_vector and t.text.lower() not in FRENCH_STOPWORDS]
        return np.mean(vs, axis=0) if vs else np.zeros(self.dim)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        return l2_normalize(np.vstack([self._vec(d) for d in self.nlp.pipe(list(texts))]))


# --------------------------------------------------------------------------------------- LSA
class LSABackend(EmbeddingBackend):
    name = "lsa"
    kind = "Sémantique latente (TF-IDF + SVD)"

    def __init__(self, n_components: int = 40):
        self.n_components = n_components
        self.label = "LSA · TF-IDF + SVD (hors-ligne)"
        self.description = "Espace sémantique latent appris sur le corpus du projet ; aucune ressource externe."
        self._vec = None
        self._svd = None

    def fit(self, corpus: Sequence[str]) -> "LSABackend":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(
            lowercase=True, strip_accents="unicode", ngram_range=(1, 2), sublinear_tf=True,
            stop_words=sorted({_strip_accents(w) for w in FRENCH_STOPWORDS}), token_pattern=r"(?u)\b[^\W\d_]{2,}\b",
        )
        X = self._vec.fit_transform(list(corpus))
        k = max(2, min(self.n_components, X.shape[1] - 1, X.shape[0] - 1))
        self._svd = TruncatedSVD(n_components=k, random_state=0).fit(X)
        return self

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        if self._vec is None:
            raise RuntimeError("LSABackend.fit(corpus) doit être appelé avant encode().")
        return l2_normalize(self._svd.transform(self._vec.transform(list(texts))))


# --------------------------------------------------------------------------------------- factory
BACKEND_CLASSES = {"sbert": SBERTBackend, "spacy": SpacyBackend, "lsa": LSABackend}
BACKEND_ORDER = ["sbert", "spacy", "lsa"]


def load_backend(name: str = "auto") -> tuple[EmbeddingBackend, list[str]]:
    """Charge un backend ; ``auto`` essaie SBERT puis, à défaut, LSA.

    spaCy n'est volontairement jamais choisi automatiquement : les vecteurs de mots moyennés servent de
    *baseline* de comparaison (cf. evaluate.py) et ont donné de moins bons classements que LSA ici.

    Retourne ``(backend, notes)`` où ``notes`` liste les replis effectués (à afficher à l'utilisateur).
    """
    notes: list[str] = []
    candidates = ["sbert", "lsa"] if name == "auto" else [name]
    for cand in candidates:
        try:
            return BACKEND_CLASSES[cand](), notes
        except Exception as exc:  # dépendance absente, modèle non téléchargeable, hors-ligne…
            notes.append(f"Backend « {cand} » indisponible ({type(exc).__name__}: {str(exc)[:120]}).")
    if name != "auto":
        notes.append("Repli sur le backend hors-ligne LSA.")
        return LSABackend(), notes
    raise RuntimeError("Aucun backend d'embeddings disponible.")  # ne devrait pas arriver (LSA = sklearn)

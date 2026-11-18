"""Moteur sémantique du recommandeur de vins (aucune dépendance à Streamlit : testable seul).

Architecture (calquée sur l'exemple « blocs de compétences → profils de poste » du sujet) :

    Référentiel                     Entrées utilisateur                    Sorties
    ───────────                     ───────────────────                    ───────
    9 blocs de goût (phrases)  ──►  texte libre / occasion / cases   ──►   profil synthétisé
    60 vins (3 champs encodés)      repas (type + texte)                   affinité sémantique (0-100 %)
    5 axes de structure             échelles de Likert (5 axes)            score de COUVERTURE par blocs
    6 « cartes de style »           notes sur des styles de vin            classement + explications
                                    budget / catégories / pays (filtres)

Tous les scores textuels sont des similarités cosinus entre embeddings ; ils sont ramenés sur [0, 1]
par normalisation min-max sur le catalogue (score *relatif* aux 60 références) avant agrégation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from embeddings import EmbeddingBackend, l2_normalize

DATA_DIR = Path(__file__).resolve().parent / "data"

# --- Paramètres du scoring (documentés et justifiés dans le README) ---------------------------------
DEFAULT_WEIGHTS = {"text": 0.30, "coverage": 0.25, "food": 0.20, "structure": 0.15, "pref": 0.10}
COMPONENT_LABELS = {
    "text": "Affinité sémantique (texte libre)",
    "coverage": "Couverture des blocs de goût",
    "food": "Accord mets-vins",
    "structure": "Structure (échelles de Likert)",
    "pref": "Préférences de style (notes)",
}
TOPK_PHRASES = 2          # score d'un bloc = moyenne des 2 phrases du bloc les plus proches du vin
BLOCK_MIX = 0.5           # poids du signal sémantique vs. cases cochées dans l'importance des blocs
BLOCK_FLOOR = 0.05        # importance minimale d'un bloc (évite un dénominateur nul)
AVOID_GAMMA = 0.30        # intensité de la pénalité pour les notes « à éviter »
PROFILE_W_DESC = 0.4      # profil du vin = 0.4·description + 0.6·profil sensoriel & mots-clés
N_CLUSTERS = 6
LEVELS = [(0.70, "Très bonne affinité"), (0.50, "Bonne affinité"), (0.0, "Affinité partielle")]  # cf. Étape 5 du sujet
FOREIGN = ["Italie", "Espagne", "Argentine", "Australie", "États-Unis", "Nouvelle-Zélande", "Allemagne", "Portugal", "Autriche"]


def level_of(score: float) -> str:
    for thr, name in LEVELS:
        if score >= thr:
            return name
    return LEVELS[-1][1]


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = float(x.min()), float(x.max())
    return np.full_like(x, 0.5) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


def country_of(region: str) -> str:
    for c in FOREIGN:
        if c in region:
            return c
    return "France"


# ============================================================================================ données
@dataclass
class Reference:
    blocks: list[dict]
    axes: dict
    style_cards: list[dict]
    occasions: dict
    food_types: dict
    guided: list[dict]

    @classmethod
    def load(cls, path: Path | str = DATA_DIR / "reference.json") -> "Reference":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(raw["blocks"], raw["axes"], raw["style_cards"], raw["occasions"], raw["food_types"], raw["guided_questions"])

    @property
    def block_ids(self) -> list[str]:
        return [b["id"] for b in self.blocks]

    @property
    def block_names(self) -> list[str]:
        return [b["name"] for b in self.blocks]


def load_wines(path: Path | str = DATA_DIR / "wines.csv") -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    required = {"WineID", "Name", "Category", "Region", "Grape", "PriceEUR", "Description", "SensoryProfile", "FoodPairing", "Keywords"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {sorted(missing)}")
    df["Country"] = df["Region"].map(country_of)
    return df


@dataclass
class UserInput:
    """Réponses du questionnaire. Tout champ vide est simplement ignoré par le scoring."""
    free_text: str = ""                                   # question ouverte : goûts / occasion
    occasion: str = ""                                    # choix unique (clé de reference.occasions)
    liked_blocks: list[str] = field(default_factory=list)  # cases à cocher : familles aromatiques aimées (ids de blocs)
    avoided_blocks: list[str] = field(default_factory=list)  # choix multiple : notes à éviter
    guided: dict[str, int] = field(default_factory=dict)   # questions guidées : id → index d'option (0 jamais, 1 aimé, 2 pas aimé)
    food_type: str = ""                                   # choix unique (clé de reference.food_types)
    food_text: str = ""                                   # question ouverte : le repas
    likert: dict[str, int] = field(default_factory=dict)  # axe → 1..5 (axes renseignés uniquement)
    style_ratings: dict[str, int] = field(default_factory=dict)  # carte → 1..5 (3 = neutre)
    budget_max: float | None = None
    categories: list[str] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    weights: dict[str, float] | None = None               # surcharge des pondérations de composantes

    def to_record(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class Result:
    table: pd.DataFrame
    present: list[str]
    weights: dict[str, float]
    block_weights: np.ndarray | None
    taste_text: str
    food_text: str
    summary: str
    n_total: int
    n_candidates: int
    user_vec: np.ndarray | None
    avoided: list[str]
    liked: list[str]
    struct_targets: dict[str, float]
    warnings: list[str] = field(default_factory=list)


# ============================================================================================ moteur
class WineEngine:
    def __init__(self, wines: pd.DataFrame, ref: Reference, backend: EmbeddingBackend):
        self.wines = wines.reset_index(drop=True)
        self.ref = ref
        self.backend = backend
        self.n = len(self.wines)
        self._build_texts()
        backend.fit(self._corpus())
        self._embed_catalogue()
        self._block_scores()
        self._axis_scores()
        self._cluster_and_project()

    # ------------------------------------------------------------------ construction du catalogue
    def _build_texts(self) -> None:
        w = self.wines
        self.t_desc = [f"Vin {c.lower()}. {d}" for c, d in zip(w["Category"], w["Description"])]
        self.t_sens = [f"{s}. {k}" for s, k in zip(w["SensoryProfile"], w["Keywords"])]
        self.t_food = list(w["FoodPairing"])

    def _corpus(self) -> list[str]:
        """Corpus d'apprentissage (utile uniquement aux backends non pré-entraînés, ex. LSA)."""
        r = self.ref
        c = self.t_desc + self.t_sens + self.t_food
        c += [p for b in r.blocks for p in b["phrases"]]
        c += [p for a in r.axes.values() for p in a["low_phrases"] + a["high_phrases"]]
        c += [s["text"] for s in r.style_cards] + list(r.occasions.values()) + list(r.food_types.values())
        return c

    def _embed_catalogue(self) -> None:
        enc = self.backend.encode
        self.E_desc, self.E_sens, self.E_food = enc(self.t_desc), enc(self.t_sens), enc(self.t_food)
        self.E_profile = l2_normalize(PROFILE_W_DESC * self.E_desc + (1 - PROFILE_W_DESC) * self.E_sens)
        self.phrases, pb = [], []
        for bi, b in enumerate(self.ref.blocks):
            for p in b["phrases"]:
                self.phrases.append(p)
                pb.append(bi)
        self.phrase_block = np.array(pb)
        self.E_phrases = enc(self.phrases)
        self.E_cards = enc([c["text"] for c in self.ref.style_cards])

    def _topk_by_block(self, cos: np.ndarray) -> np.ndarray:
        """cos : (m, n_phrases) → (m, n_blocks), moyenne des TOPK_PHRASES meilleures phrases de chaque bloc."""
        cols = []
        for bi in range(len(self.ref.blocks)):
            sub = cos[:, self.phrase_block == bi]
            k = min(TOPK_PHRASES, sub.shape[1])
            cols.append(np.sort(sub, axis=1)[:, -k:].mean(axis=1))
        return np.column_stack(cols)

    def _block_scores(self) -> None:
        self.phrase_cos = self.E_profile @ self.E_phrases.T               # (n_vins, n_phrases)
        self.S_block_raw = self._topk_by_block(self.phrase_cos)           # (n_vins, n_blocs)
        self.S_block = np.column_stack([minmax(self.S_block_raw[:, j]) for j in range(self.S_block_raw.shape[1])])

    def _axis_scores(self) -> None:
        """Position de chaque vin sur les 5 axes de structure, par proximité sémantique aux pôles (zéro-shot).

        d(vin) = moyenne des cosinus avec les phrases du pôle « haut » − moyenne avec celles du pôle « bas »,
        puis mise à l'échelle robuste [P5, P95] → [0, 1] sur le catalogue. Aucune note de structure n'est
        saisie à la main dans le jeu de données : tout vient du texte (cf. tests de cohérence dans le README).
        """
        self.axis_keys = list(self.ref.axes)
        scaled, raws = [], []
        for key in self.axis_keys:
            ax = self.ref.axes[key]
            hi, lo = self.backend.encode(ax["high_phrases"]), self.backend.encode(ax["low_phrases"])
            d = (self.E_profile @ hi.T).mean(axis=1) - (self.E_profile @ lo.T).mean(axis=1)
            p5, p95 = np.percentile(d, 5), np.percentile(d, 95)
            scaled.append(np.clip((d - p5) / (p95 - p5), 0, 1) if p95 - p5 > 1e-9 else np.full(self.n, 0.5))
            raws.append(d)
        self.axis_pct = np.column_stack(scaled)    # (n_vins, 5) dans [0, 1] : position sur l'axe
        self.axis_raw = np.column_stack(raws)

    def _cluster_and_project(self) -> None:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        self.pca = PCA(n_components=2, random_state=0).fit(self.E_profile)
        self.xy = self.pca.transform(self.E_profile)
        k = min(N_CLUSTERS, self.n)
        self.cluster = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(self.E_profile)
        plural = {"Rouge": "Rouges", "Blanc": "Blancs", "Rosé": "Rosés", "Effervescent": "Effervescents", "Doux / Liquoreux": "Doux"}
        global_mean = self.S_block.mean(axis=0)
        cats = self.wines["Category"].to_numpy()
        names = []
        for c in range(k):
            members = self.cluster == c
            top = np.argsort(-(self.S_block[members].mean(axis=0) - global_mean))[:2]   # blocs *distinctifs* du cluster
            cat, share = pd.Series(cats[members]).value_counts().pipe(lambda v: (v.index[0], v.iloc[0] / v.sum()))
            prefix = plural.get(cat, cat) if share >= 0.6 else "Mixte"
            names.append(f"C{c + 1} · {prefix} — " + " + ".join(self.ref.block_names[i] for i in top))
        self.cluster_names = names

    # ------------------------------------------------------------------ accès catalogue
    @property
    def categories(self) -> list[str]:
        return sorted(self.wines["Category"].unique())

    @property
    def countries(self) -> list[str]:
        return sorted(self.wines["Country"].unique(), key=lambda c: (c != "France", c))

    @property
    def price_bounds(self) -> tuple[int, int]:
        return int(self.wines["PriceEUR"].min()), int(np.ceil(self.wines["PriceEUR"].max()))

    def _enc1(self, text: str) -> np.ndarray | None:
        v = self.backend.encode([text])[0]
        return v if np.linalg.norm(v) > 1e-9 else None

    # ------------------------------------------------------------------ interprétation des réponses
    def resolve_blocks(self, inp: UserInput) -> tuple[list[str], list[str]]:
        """Fusionne cases cochées, « à éviter » et questions guidées en (blocs aimés, blocs à éviter)."""
        liked, avoided = list(inp.liked_blocks), list(inp.avoided_blocks)
        for q in self.ref.guided:
            ans = inp.guided.get(q["id"], 0)
            if ans == 1 and q["block"] not in liked:
                liked.append(q["block"])
            elif ans == 2 and q["block"] not in avoided:
                avoided.append(q["block"])
        avoided = [b for b in avoided if b not in liked]   # une case « j'aime » explicite l'emporte
        return liked, avoided

    def semantic_text(self, inp: UserInput) -> str:
        parts = []
        if inp.free_text.strip():
            parts.append(inp.free_text.strip())
        if inp.occasion in self.ref.occasions:
            parts.append(self.ref.occasions[inp.occasion])
        return ". ".join(parts)

    def taste_text(self, inp: UserInput) -> str:
        parts = [self.semantic_text(inp)] if self.semantic_text(inp) else []
        liked, _ = self.resolve_blocks(inp)
        bmap = {b["id"]: b for b in self.ref.blocks}
        parts += [", ".join(bmap[b]["phrases"][:2]) for b in liked if b in bmap]
        return ". ".join(parts)

    def food_text(self, inp: UserInput) -> str:
        parts = []
        if inp.food_type in self.ref.food_types:
            parts.append(self.ref.food_types[inp.food_type])
        if inp.food_text.strip():
            parts.append(inp.food_text.strip())
        return ". ".join(parts)

    def block_weights(self, inp: UserInput) -> np.ndarray | None:
        """Importance W_b de chaque bloc pour l'utilisateur (≙ poids Wi de la formule de couverture)."""
        liked, _ = self.resolve_blocks(inp)
        nb = len(self.ref.blocks)
        explicit = np.zeros(nb)
        for b in liked:
            explicit[self.ref.block_ids.index(b)] = 1.0
        sem = None
        text = self.semantic_text(inp)
        if text:
            v = self._enc1(text)
            if v is not None:
                sem = minmax(self._topk_by_block((self.E_phrases @ v)[None, :])[0]) ** 2
        if sem is not None and liked:
            w = BLOCK_MIX * sem + (1 - BLOCK_MIX) * explicit
        elif sem is not None:
            w = sem
        elif liked:
            w = explicit
        else:
            return None
        return np.maximum(w, BLOCK_FLOOR)

    # ------------------------------------------------------------------ recommandation
    def recommend(self, inp: UserInput, top_n: int | None = None) -> Result:
        warnings: list[str] = []
        comps: dict[str, np.ndarray] = {}
        raw: dict[str, np.ndarray] = {}
        liked, avoided = self.resolve_blocks(inp)

        taste = self.taste_text(inp)
        uvec = None
        if taste:
            uvec = self._enc1(taste)
            if uvec is None:
                warnings.append("Aucun mot de votre description n'est connu du modèle d'embeddings actuel : essayez un autre backend ou reformulez.")
            else:
                raw["text"] = self.E_profile @ uvec
                comps["text"] = minmax(raw["text"])
        W = self.block_weights(inp)
        if W is not None:
            comps["coverage"] = (self.S_block * W).sum(axis=1) / W.sum()      # Coverage = Σ Wi·Si / Σ Wi

        food = self.food_text(inp)
        if food:
            fv = self._enc1(food)
            if fv is None:
                warnings.append("Le contexte de repas n'a pas pu être analysé par le modèle actuel.")
            else:
                raw["food"] = self.E_food @ fv
                comps["food"] = minmax(raw["food"])

        targets = {k: (v - 1) / 4 for k, v in inp.likert.items() if k in self.axis_keys}
        if targets:
            cols = [self.axis_keys.index(k) for k in targets]
            t = np.array(list(targets.values()))
            comps["structure"] = 1 - np.abs(self.axis_pct[:, cols] - t).mean(axis=1)

        rated = {cid: r - 3 for cid, r in inp.style_ratings.items() if r != 3}
        if rated:
            ids = [c["id"] for c in self.ref.style_cards]
            idx = [ids.index(c) for c in rated]
            wts = np.array(list(rated.values()), dtype=float)
            s = (self.E_profile @ self.E_cards[idx].T) @ wts / np.abs(wts).sum()
            comps["pref"] = minmax(s)

        present = [c for c in DEFAULT_WEIGHTS if c in comps]
        base_w = {**DEFAULT_WEIGHTS, **(inp.weights or {})}
        tot = sum(base_w[c] for c in present)
        eff = {c: (base_w[c] / tot if tot > 0 else 1 / len(present)) for c in present}

        mask = np.ones(self.n, dtype=bool)
        if inp.budget_max is not None:
            mask &= (self.wines["PriceEUR"] <= inp.budget_max).to_numpy()
        if inp.categories:
            mask &= self.wines["Category"].isin(inp.categories).to_numpy()
        if inp.countries:
            mask &= self.wines["Country"].isin(inp.countries).to_numpy()

        summary = self._summary(inp, W, liked, avoided, targets)
        empty = Result(pd.DataFrame(), present, eff, W, taste, food, summary, self.n, int(mask.sum()), uvec, avoided, liked, targets, warnings)
        if not present:
            return empty

        affinity = sum(eff[c] * comps[c] for c in present)
        av_idx = [self.ref.block_ids.index(b) for b in avoided]
        penalty = self.S_block[:, av_idx].mean(axis=1) if av_idx else np.zeros(self.n)
        final = np.clip(affinity - AVOID_GAMMA * penalty, 0, 1)

        df = self.wines[["WineID", "Name", "Category", "Region", "Country", "Grape", "PriceEUR"]].copy()
        df["idx"] = np.arange(self.n)
        df["Score"] = final
        df["Level"] = [level_of(s) for s in final]
        for c in DEFAULT_WEIGHTS:
            df[f"s_{c}"] = comps.get(c, np.full(self.n, np.nan))
            df[f"contrib_{c}"] = eff[c] * comps[c] if c in comps else 0.0
        df["Penalty"] = penalty
        df["contrib_penalty"] = -AVOID_GAMMA * penalty
        df["cos_text"] = raw.get("text", np.full(self.n, np.nan))
        df["cos_food"] = raw.get("food", np.full(self.n, np.nan))
        df = df[mask].sort_values(["Score", "PriceEUR"], ascending=[False, True]).reset_index(drop=True)
        df.insert(0, "Rank", np.arange(1, len(df) + 1))
        if top_n:
            df = df.head(top_n)
        return Result(df, present, eff, W, taste, food, summary, self.n, int(mask.sum()), uvec, avoided, liked, targets, warnings)

    # ------------------------------------------------------------------ explications & synthèse
    def _summary(self, inp, W, liked, avoided, targets) -> str:
        parts = []
        names = self.ref.block_names
        if W is not None:
            order = np.argsort(-W)
            top = [names[i] for i in order[:3] if W[i] >= 0.5 * W.max() and W[i] > BLOCK_FLOOR * 2]
            if top:
                parts.append("dominante « " + " », « ".join(top) + " »")
        style = []
        for k, t in targets.items():
            ax = self.ref.axes[k]
            if t >= 0.75:
                style.append(ax["high_word"])
            elif t <= 0.25:
                style.append(ax["low_word"])
        if style:
            parts.append("style " + ", ".join(style))
        if avoided:
            bn = dict(zip(self.ref.block_ids, names))
            parts.append("à éviter : " + ", ".join(bn[b] for b in avoided))
        ctx = inp.food_type or ("repas décrit librement" if inp.food_text.strip() else "")
        if ctx:
            parts.append(f"repas : {ctx.lower()}")
        if inp.occasion:
            parts.append(f"occasion : {inp.occasion.lower()}")
        if inp.budget_max is not None:
            parts.append(f"budget ≤ {inp.budget_max:.0f} €")
        return "Profil détecté — " + " ; ".join(parts) + "." if parts else "Profil peu renseigné : ajoutez des préférences pour affiner."

    def explain(self, res: Result, idx: int) -> list[str]:
        """Phrases d'explication pour le vin d'indice `idx` (position dans le catalogue)."""
        out: list[str] = []
        names = self.ref.block_names
        row = res.table[res.table["idx"] == idx]
        if row.empty:
            return out
        row = row.iloc[0]
        if res.block_weights is not None:
            contrib = res.block_weights * self.S_block[idx]
            for b in np.argsort(-contrib)[:2]:
                if self.S_block[idx, b] < 0.4:
                    continue
                cols = np.where(self.phrase_block == b)[0]
                best = self.phrases[cols[np.argmax(self.phrase_cos[idx, cols])]]
                out.append(f"Bloc « {names[b]} » : couverture {self.S_block[idx, b]:.0%} (importance pour vous {res.block_weights[b] / res.block_weights.max():.0%}) — notes proches : « {best} ».")
        if "food" in res.present and not np.isnan(row["cos_food"]):
            out.append(f"Accord mets-vins : cosinus {row['cos_food']:.2f} entre votre repas et « {self.wines.loc[idx, 'FoodPairing']} ».")
        if "text" in res.present and not np.isnan(row["cos_text"]):
            out.append(f"Texte libre : cosinus {row['cos_text']:.2f} avec le profil gustatif du vin.")
        for k, t in res.struct_targets.items():
            wine_pos = self.axis_pct[idx, self.axis_keys.index(k)] * 4 + 1
            out.append(f"{self.ref.axes[k]['label'].split(' (')[0]} : souhaité {t * 4 + 1:.0f}/5, profil du vin ≈ {wine_pos:.1f}/5.")
        if row["Penalty"] > 0.5 and res.avoided:
            bn = dict(zip(self.ref.block_ids, names))
            out.append("Attention : ce vin présente des notes proches de ce que vous souhaitez éviter (" + ", ".join(bn[b] for b in res.avoided) + ").")
        return out

    # ------------------------------------------------------------------ visualisations
    def map_frame(self, user_vec: np.ndarray | None = None) -> tuple[pd.DataFrame, tuple[float, float] | None]:
        df = self.wines[["WineID", "Name", "Category", "Region", "PriceEUR"]].copy()
        df["x"], df["y"] = self.xy[:, 0], self.xy[:, 1]
        df["Cluster"] = [self.cluster_names[c] for c in self.cluster]
        pos = None
        if user_vec is not None:
            p = self.pca.transform(user_vec[None, :])[0]
            pos = (float(p[0]), float(p[1]))
        return df, pos

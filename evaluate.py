"""Évaluation comparative des backends d'embeddings sur 19 requêtes-tests (13 directes + 6 paraphrases).

    python evaluate.py              # tous les backends disponibles (sbert, spacy, lsa) + baseline mots-clés
    python evaluate.py sbert lsa    # sous-ensemble

Pour chaque requête, un jeu de vins « pertinents » a été défini à la main (connaissance œnologique de base)
AVANT de lancer les calculs. C'est un test de cohérence, pas un benchmark : 19 requêtes, annotateur unique.
Métriques : P@5 (précision des 5 premiers), Hit@1 (le 1er résultat est pertinent), MRR (rang du 1er pertinent).
Écrit aussi docs/evaluation_results.md.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from embeddings import load_backend  # noqa: E402
from wine_engine import Reference, UserInput, WineEngine, load_wines  # noqa: E402

DIRECT: list[tuple[str, set[str]]] = [
    ("un vin blanc vif et minéral pour accompagner des huîtres",
     {"Chablis", "Muscadet Sèvre-et-Maine sur lie", "Picpoul de Pinet", "Albariño Rías Baixas", "Champagne blanc de blancs", "Champagne brut"}),
    ("un rouge puissant et tannique pour une côte de bœuf grillée",
     {"Saint-Estèphe", "Pauillac", "Cahors", "Barolo", "Bandol rouge", "Châteauneuf-du-Pape", "Shiraz de Barossa", "Malbec de Mendoza"}),
    ("un vin doux pour accompagner un foie gras",
     {"Sauternes", "Monbazillac", "Gewurztraminer d'Alsace", "Pinot gris d'Alsace"}),
    ("un rosé frais et fruité pour un apéritif d'été",
     {"Côtes de Provence rosé", "Sancerre rosé", "Rosé d'Anjou", "Bandol rosé"}),
    ("un vin léger et fruité, facile à boire entre amis, sur la fraise et la cerise",
     {"Beaujolais-Villages", "Bourgogne Pinot noir", "Chinon", "Morgon", "Côtes du Rhône"}),
    ("un vin aromatique sur le litchi et la rose pour un plat asiatique épicé",
     {"Gewurztraminer d'Alsace", "Riesling d'Alsace", "Riesling Mosel Kabinett", "Pinot gris d'Alsace", "Grüner Veltliner"}),
    ("des bulles fines pour célébrer un repas de fête",
     {"Champagne brut", "Champagne blanc de blancs", "Champagne rosé", "Crémant d'Alsace", "Crémant de Loire", "Prosecco", "Cava"}),
    ("un vin boisé, vanillé et rond avec du beurre",
     {"Meursault", "Chardonnay de Californie", "Rioja Reserva", "Pouilly-Fuissé"}),
    ("un rouge poivré avec des notes de garrigue pour un barbecue entre amis",
     {"Côtes du Rhône", "Crozes-Hermitage", "Châteauneuf-du-Pape", "Pic Saint-Loup", "Bandol rouge"}),
    ("un blanc citronné et herbacé pour une salade et du fromage de chèvre",
     {"Sancerre blanc", "Pouilly-Fumé", "Sauvignon blanc de Marlborough", "Grüner Veltliner", "Vouvray sec"}),
    ("un vin doux et léger pour accompagner une tarte aux fruits",
     {"Moscato d'Asti", "Monbazillac", "Sauternes", "Riesling Mosel Kabinett"}),
    ("un vin de garde très complexe, avec du cassis et du cèdre, pour une grande occasion",
     {"Pauillac", "Saint-Estèphe", "Pomerol", "Barolo", "Saint-Émilion grand cru"}),
    ("un vin salin et iodé pour des moules et des crevettes",
     {"Muscadet Sèvre-et-Maine sur lie", "Picpoul de Pinet", "Albariño Rías Baixas", "Chablis"}),
]

# Paraphrases : volontairement éloignées du vocabulaire du jeu de données (synonymes, périphrases, registre courant).
# Elles mesurent la capacité à généraliser *sémantiquement* au-delà du recouvrement de mots-clés.
PARAPHRASES: list[tuple[str, set[str]]] = [
    ("un vin qui sent la mer, parfait avec des coquillages",
     {"Muscadet Sèvre-et-Maine sur lie", "Picpoul de Pinet", "Chablis", "Albariño Rías Baixas", "Champagne blanc de blancs"}),
    ("un rouge costaud qui laisse la bouche sèche, pour du gibier",
     {"Saint-Estèphe", "Pauillac", "Cahors", "Barolo", "Bandol rouge", "Gevrey-Chambertin", "Côte-Rôtie"}),
    ("un vin tout sucré avec un goût de confiture d'abricot, pour le dessert",
     {"Sauternes", "Monbazillac", "Moscato d'Asti"}),
    ("une boisson pétillante pour trinquer avec des invités",
     {"Champagne brut", "Champagne blanc de blancs", "Champagne rosé", "Crémant d'Alsace", "Crémant de Loire", "Prosecco", "Cava"}),
    ("un vin tendre et facile pour quelqu'un qui découvre le vin",
     {"Rosé d'Anjou", "Beaujolais-Villages", "Moscato d'Asti", "Prosecco", "Bordeaux rouge", "Mâcon-Villages", "Côtes du Rhône"}),
    ("un vin qui sent les fleurs et les fruits exotiques, très onctueux",
     {"Gewurztraminer d'Alsace", "Condrieu", "Pinot gris d'Alsace"}),
]
QUERIES = [("directes", q, r) for q, r in DIRECT] + [("paraphrases", q, r) for q, r in PARAPHRASES]

# Vérité terrain pour les axes de structure (connaissance générale, indépendante du texte du jeu de données)
AXIS_TRUTH = {
    "body": (["Pauillac", "Saint-Estèphe", "Barolo", "Shiraz de Barossa", "Châteauneuf-du-Pape", "Cahors", "Bandol rouge", "Pomerol", "Arbois Vin jaune", "Meursault"],
             ["Muscadet Sèvre-et-Maine sur lie", "Beaujolais-Villages", "Riesling Mosel Kabinett", "Prosecco", "Moscato d'Asti", "Picpoul de Pinet", "Mâcon-Villages", "Bourgogne Pinot noir", "Chinon", "Rosé d'Anjou"]),
    "acidity": (["Chablis", "Muscadet Sèvre-et-Maine sur lie", "Sancerre blanc", "Riesling d'Alsace", "Picpoul de Pinet", "Champagne blanc de blancs", "Chianti Classico", "Grüner Veltliner", "Sauvignon blanc de Marlborough", "Albariño Rías Baixas"],
                ["Condrieu", "Gewurztraminer d'Alsace", "Meursault", "Chardonnay de Californie", "Pomerol", "Malbec de Mendoza", "Pinot gris d'Alsace", "Porto Tawny", "Shiraz de Barossa", "Saint-Émilion grand cru"]),
}


def ranking(eng: WineEngine, inp: UserInput) -> list[str]:
    """Noms classés ; liste vide si aucun mot de la requête n'est connu du modèle (cas LSA / spaCy hors vocabulaire)."""
    t = eng.recommend(inp).table
    return list(t["Name"]) if len(t) else []


def metrics(ranked: list[str], rel: set[str]) -> tuple[float, float, float]:
    if not ranked:
        return 0.0, 0.0, 0.0
    p5 = len(set(ranked[:5]) & rel) / 5
    first = next((i for i, n in enumerate(ranked, 1) if n in rel), None)
    return p5, float(ranked[0] in rel), (1 / first if first else 0.0)


def auc(score: np.ndarray, pos: list[int], neg: list[int]) -> float:
    return float(np.mean([(score[p] > score[n]) + 0.5 * (score[p] == score[n]) for p in pos for n in neg]))


def keyword_baseline(wines: pd.DataFrame):
    """Baseline « mots-clés » : TF-IDF brut, sans espace sémantique latent."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = (wines["Category"] + " " + wines["Description"] + " " + wines["SensoryProfile"] + " " + wines["FoodPairing"] + " " + wines["Keywords"]).tolist()
    vec = TfidfVectorizer(strip_accents="unicode", sublinear_tf=True).fit(docs)
    M = vec.transform(docs)
    return lambda q: np.asarray((M @ vec.transform([q]).T).todense()).ravel()


def main(names: list[str]) -> None:
    wines, ref = load_wines(), Reference.load()
    rows, notes, axis_rows = [], [], []

    kw = keyword_baseline(wines)
    for grp, q, rel in QUERIES:
        order = np.argsort(-kw(q))
        p5, h1, mrr = metrics([wines.Name[i] for i in order], rel)
        rows.append(("Mots-clés (TF-IDF brut, baseline)", "—", grp, q, p5, h1, mrr))

    for name in names:
        backend, n = load_backend(name)
        if name != "auto" and backend.name != name:
            notes.append(f"`{name}` indisponible → ignoré ({'; '.join(n)})")
            continue
        eng = WineEngine(wines, ref, backend)
        for grp, q, rel in QUERIES:
            for mode in ("texte", "texte+repas"):
                inp = UserInput(free_text=q, food_text=q if mode == "texte+repas" else "")
                ranked = ranking(eng, inp)
                rows.append((backend.label, mode, grp, q, *metrics(ranked, rel)))
        idx = {n_: i for i, n_ in enumerate(eng.wines.Name)}
        cat = eng.wines["Category"].to_numpy()
        truth = {**{k: ([idx[x] for x in p], [idx[x] for x in n_]) for k, (p, n_) in AXIS_TRUTH.items()},
                 "sweetness": (list(np.where(cat == "Doux / Liquoreux")[0]), list(np.where(cat != "Doux / Liquoreux")[0])),
                 "tannin": ([idx[x] for x in ["Saint-Estèphe", "Pauillac", "Cahors", "Barolo", "Bandol rouge", "Shiraz de Barossa"]],
                            sorted(set(np.where(cat != "Rouge")[0]) | {idx["Beaujolais-Villages"], idx["Bourgogne Pinot noir"]}))}
        axis_rows.append({"Backend": backend.label, **{k: round(auc(eng.axis_pct[:, eng.axis_keys.index(k)], *truth[k]), 2) for k in truth}})

    df = pd.DataFrame(rows, columns=["Backend", "Mode", "Groupe", "Requête", "P@5", "Hit@1", "MRR"])
    summary = df.groupby(["Backend", "Mode", "Groupe"], sort=False)[["P@5", "Hit@1", "MRR"]].mean().round(3).reset_index()
    print("\n=== Moyennes (13 requêtes directes, 6 paraphrases) ===")
    print(summary.to_string(index=False))
    axes = pd.DataFrame(axis_rows)
    if len(axes):
        print("\n=== Cohérence des axes de structure (AUC : 0,5 = hasard, 1 = parfait) ===")
        print(axes.to_string(index=False))
    for n in notes:
        print("NOTE:", n)

    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    md = ["# Résultats d'évaluation", "", f"{len(QUERIES)} requêtes-tests en français (13 directes + 6 paraphrases), jeux de vins pertinents annotés à la main (annotateur unique) : test de cohérence, pas benchmark.", "",
          "## Qualité du classement", "", summary.to_markdown(index=False), "", "## Cohérence des axes de structure (AUC)", ""]
    md += [axes.to_markdown(index=False) if len(axes) else "_non calculé_", ""]
    md += ["## Détail par requête", "", df.round(2).to_markdown(index=False), ""]
    md += [f"> {n}" for n in notes]
    (out / "evaluation_results.md").write_text("\n".join(md), encoding="utf-8")
    print("\nÉcrit :", out / "evaluation_results.md")


if __name__ == "__main__":
    main(sys.argv[1:] or ["sbert", "spacy", "lsa"])

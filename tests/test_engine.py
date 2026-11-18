"""Vérifications numériques du moteur (sans pytest) :  python tests/test_engine.py [backend]"""
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embeddings import load_backend  # noqa: E402
from wine_engine import AVOID_GAMMA, DEFAULT_WEIGHTS, Reference, UserInput, WineEngine, load_wines  # noqa: E402

backend, _ = load_backend(sys.argv[1] if len(sys.argv) > 1 else "lsa")
eng = WineEngine(load_wines(), Reference.load(), backend)
print("backend :", backend.label)

# 0) dataset conforme au sujet
w = eng.wines
assert len(w) >= 50, "minimum recommandé : 50 références"
assert {"WineID", "Category", "Region", "Description", "SensoryProfile", "FoodPairing", "Keywords"} <= set(w.columns)
assert w["WineID"].is_unique and not w.isna().any().any()

base = UserInput(free_text="un vin frais et fruité sur la fraise et la cerise", liked_blocks=["B1"], food_type="Charcuterie / apéritif",
                 likert={"body": 2, "acidity": 4}, style_ratings={"S4": 5, "S2": 1})
r = eng.recommend(base)
t = r.table

# 1) formule de couverture : Σ Wi·Si / Σ Wi recalculée à la main pour le 1er vin
i = int(t.iloc[0]["idx"])
manual = float((r.block_weights * eng.S_block[i]).sum() / r.block_weights.sum())
assert abs(manual - t.iloc[0]["s_coverage"]) < 1e-12, (manual, t.iloc[0]["s_coverage"])

# 2) poids effectifs renormalisés (somme = 1) et score = Σ ω·s − γ·pénalité, borné dans [0, 1]
assert abs(sum(r.weights.values()) - 1) < 1e-12 and set(r.weights) == set(r.present)
row = t.iloc[0]
affinity = sum(r.weights[c] * row[f"s_{c}"] for c in r.present)
assert abs(row["Score"] - np.clip(affinity - AVOID_GAMMA * row["Penalty"], 0, 1)) < 1e-12
assert t["Score"].between(0, 1).all() and t["Score"].is_monotonic_decreasing

# 3) composantes absentes ignorées : sans repas ni Likert ni notes, seules text+coverage comptent
r2 = eng.recommend(UserInput(free_text="vin rouge puissant", liked_blocks=["B9"]))
assert r2.present == ["text", "coverage"], r2.present

# 4) filtres durs respectés
r3 = eng.recommend(UserInput(free_text="vin frais", budget_max=15, categories=["Blanc"], countries=["France"]))
assert len(r3.table) and (r3.table["PriceEUR"] <= 15).all() and (r3.table["Category"] == "Blanc").all() and (r3.table["Country"] == "France").all()

# 5) pénalité : éviter un bloc ne peut qu'abaisser le score des vins qui l'expriment
r4 = eng.recommend(UserInput(**{**base.__dict__, "avoided_blocks": ["B5"]}))
a, b = t.set_index("WineID")["Score"], r4.table.set_index("WineID")["Score"]
assert (b <= a.reindex(b.index) + 1e-12).all() and (b < a.reindex(b.index) - 1e-6).any()

# 6) entrées vides / filtres impossibles → pas de plantage
assert eng.recommend(UserInput()).table.empty and not eng.recommend(UserInput()).present
assert eng.recommend(UserInput(free_text="vin", budget_max=1)).table.empty

# 7) la pondération utilisateur change bien le classement des composantes
r5 = eng.recommend(UserInput(**{**base.__dict__, "weights": {**DEFAULT_WEIGHTS, "food": 0.0}}))
assert abs(r5.weights["food"]) < 1e-12 and abs(sum(r5.weights.values()) - 1) < 1e-12

# 8) explications disponibles pour chaque vin du top 5
for idx in t.head(5)["idx"]:
    assert eng.explain(r, int(idx)), "explication vide"
print("OK — 8 groupes de vérifications passés")

"""Recommandeur sémantique de vins — prototype pédagogique (ESME ING5 A3MSI, Thème 5).

Lancer :  streamlit run app.py
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from embeddings import load_backend
from wine_engine import (
    AVOID_GAMMA, BLOCK_FLOOR, BLOCK_MIX, COMPONENT_LABELS, DATA_DIR, DEFAULT_WEIGHTS, LEVELS, PROFILE_W_DESC,
    TOPK_PHRASES, Reference, UserInput, WineEngine, load_wines,
)

ROOT = Path(__file__).resolve().parent
RESPONSES_CSV = ROOT / "responses" / "responses.csv"
CAT_COLORS = {"Rouge": "#8E1B3A", "Blanc": "#C9B037", "Rosé": "#E58FA5", "Effervescent": "#4F8FC0", "Doux / Liquoreux": "#C9822B"}
WINE_SCALE = "Burg"
NONE = "— non précisé —"

st.set_page_config(page_title="Recommandeur sémantique de vins", page_icon="🍷", layout="wide")

_ST_VERSION = tuple(int(x) for x in st.__version__.split(".")[:2])
STRETCH = {"width": "stretch"} if _ST_VERSION >= (1, 50) else {"use_container_width": True}


# ============================================================================================ chargement
@st.cache_resource(show_spinner="Chargement du modèle d'embeddings et encodage du catalogue…")
def get_engine(backend_name: str, data_stamp: float):
    backend, notes = load_backend(backend_name)
    return WineEngine(load_wines(), Reference.load(), backend), notes


def data_stamp() -> float:
    return max((DATA_DIR / f).stat().st_mtime for f in ("wines.csv", "reference.json"))


def log_response(inp: UserInput, res, backend_label: str) -> None:
    """Stockage structuré des réponses (CSV). Silencieux si le dossier n'est pas inscriptible."""
    try:
        RESPONSES_CSV.parent.mkdir(exist_ok=True)
        top = res.table.head(3)
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "session": st.session_state.setdefault("sid", uuid.uuid4().hex[:8]),
            "backend": backend_label,
            "answers_json": json.dumps(inp.to_record(), ensure_ascii=False),
            "top3_ids": ";".join(top["WineID"]),
            "top3_scores": ";".join(f"{s:.3f}" for s in top["Score"]),
        }
        pd.DataFrame([row]).to_csv(RESPONSES_CSV, mode="a", header=not RESPONSES_CSV.exists(), index=False, encoding="utf-8")
    except OSError:
        pass


# ============================================================================================ en-tête
st.title("🍷 Recommandeur sémantique de vins")
st.caption("Prototype pédagogique · ESME ING5 A3MSI · Thème 5 — Analyse sémantique du langage sensoriel et système de recommandation")
st.warning(
    "**Démonstration pédagogique destinée à un public adulte.** Aucun achat, aucune incitation à consommer et aucune "
    "personnalisation fondée sur des informations de santé : ne saisissez pas de données de santé ou personnelles. "
    "*L'abus d'alcool est dangereux pour la santé, à consommer avec modération.*",
    icon="⚠️",
)
st.checkbox("Je confirme avoir 18 ans ou plus et utiliser cette application uniquement à des fins pédagogiques.", key="adult_ok")

# ============================================================================================ barre latérale
with st.sidebar:
    st.header("⚙️ Moteur sémantique")
    backend_choice = st.selectbox(
        "Modèle d'embeddings",
        ["auto", "sbert", "spacy", "lsa"],
        format_func={"auto": "Automatique (SBERT, sinon LSA)", "sbert": "SBERT multilingue (recommandé)",
                     "spacy": "spaCy · vecteurs de mots (baseline)", "lsa": "LSA TF-IDF+SVD (hors-ligne)"}.get,
        help="SBERT = embeddings contextuels. spaCy = moyenne de vecteurs de mots (comparaison). LSA = repli sans téléchargement.",
        key="backend_choice",
    )
    engine, backend_notes = get_engine(backend_choice, data_stamp())
    st.success(f"Actif : **{engine.backend.label}**")
    st.caption(f"{engine.backend.kind}. {engine.backend.description}")
    for n in backend_notes:
        st.info(n, icon="ℹ️")

    st.header("⚖️ Pondérations")
    st.caption("Poids ω de chaque composante dans l'affinité finale. Les composantes non renseignées sont ignorées et les poids renormalisés.")
    weights = {c: st.slider(COMPONENT_LABELS[c], 0.0, 1.0, float(DEFAULT_WEIGHTS[c]), 0.05, key=f"w_{c}") for c in DEFAULT_WEIGHTS}
    top_n = st.slider("Nombre de recommandations", 3, 15, 5)
    st.divider()
    st.caption(f"Catalogue : **{engine.n} références** · {len(engine.ref.blocks)} blocs de goût · {len(engine.phrases)} phrases de référence.")

if not st.session_state.get("adult_ok"):
    st.info("Cochez la case de confirmation ci-dessus pour accéder au questionnaire.")
    st.stop()

ref = engine.ref

# ============================================================================================ questionnaire
st.header("1 · Questionnaire")
with st.expander("Répondez en langage naturel : plus vous décrivez, plus l'analyse sémantique est précise", expanded="inp" not in st.session_state):
    with st.form("questionnaire"):
        st.subheader("🗣️ Question ouverte — vos goûts et l'occasion")
        c1, c2 = st.columns([3, 2])
        free_text = c1.text_area(
            "Décrivez en quelques phrases le type de vin que vous aimez ou que vous recherchez",
            placeholder="Ex. : J'aime les vins frais et légers, avec des notes de fruits rouges, pour un pique-nique entre amis…",
            height=110)
        occasion = c2.radio("Occasion (choix unique)", [NONE] + list(ref.occasions), index=0)

        st.subheader("☑️ Cases à cocher — ce que vous aimez dans un vin")
        cols = st.columns(3)
        liked_blocks = [b["id"] for i, b in enumerate(ref.blocks) if cols[i % 3].checkbox(f"{b['name']}", key=f"like_{b['id']}", help=" · ".join(b["phrases"][:3]))]
        avoided_blocks = st.multiselect("Choix multiple — notes que vous préférez éviter", [b["id"] for b in ref.blocks],
                                        format_func=dict(zip(ref.block_ids, ref.block_names)).get)

        st.subheader("🧭 Questions guidées — votre expérience")
        guided = {}
        for q in ref.guided:
            ans = st.radio(q["question"], q["options"], index=0, horizontal=True, key=f"g_{q['id']}")
            guided[q["id"]] = q["options"].index(ans)

        st.subheader("🍽️ Contexte d'accord mets-vins")
        c3, c4 = st.columns([2, 3])
        food_type = c3.selectbox("Type de plat (choix unique)", [NONE] + list(ref.food_types))
        food_text = c4.text_area("Décrivez le repas (question ouverte)", placeholder="Ex. : un plateau de fruits de mer avec du citron et du pain de seigle…", height=80)

        st.subheader("📊 Échelles de Likert — votre style préféré")
        st.caption("Laissez « Indifférent » pour ignorer un critère. 1 = pôle gauche, 5 = pôle droit.")
        likert = {}
        lc = st.columns(2)
        for i, (key, ax) in enumerate(ref.axes.items()):
            with lc[i % 2]:
                v = st.select_slider(ax["label"], options=["Indifférent", 1, 2, 3, 4, 5], value="Indifférent", key=f"lk_{key}",
                                     help=f"1 = {ax['low_label']} · 5 = {ax['high_label']}")
                if v != "Indifférent":
                    likert[key] = int(v)

        st.subheader("⭐ Évaluation de préférences — notez ces descriptions de styles")
        st.caption("De 1 (« pas du tout pour moi ») à 5 (« exactement ce que j'aime »). Les notes < 3 repoussent les vins proches de la description.")
        style_ratings = {}
        for card in ref.style_cards:
            st.markdown(f"*{card['text']}*")
            v = st.select_slider(card["id"], options=["Non évalué", 1, 2, 3, 4, 5], value="Non évalué", key=f"sc_{card['id']}", label_visibility="collapsed")
            if v != "Non évalué":
                style_ratings[card["id"]] = int(v)

        st.subheader("💶 Budget et contraintes")
        lo, hi = engine.price_bounds
        c5, c6, c7 = st.columns(3)
        budget = c5.slider("Budget maximum par bouteille (€) — filtre", lo, hi, hi)
        categories = c6.multiselect("Catégories (vide = toutes)", engine.categories)
        countries = c7.multiselect("Pays / zones (vide = tous)", engine.countries)

        submitted = st.form_submit_button("🔎 Obtenir mes recommandations", type="primary")

if submitted:
    st.session_state["inp"] = UserInput(
        free_text=free_text, occasion="" if occasion == NONE else occasion, liked_blocks=liked_blocks, avoided_blocks=avoided_blocks,
        guided=guided, food_type="" if food_type == NONE else food_type, food_text=food_text, likert=likert,
        style_ratings=style_ratings, budget_max=float(budget), categories=categories, countries=countries)
    st.session_state["logged"] = False

if "inp" not in st.session_state:
    st.info("Remplissez le questionnaire puis validez : les recommandations apparaîtront ici.")
    st.stop()

# Le calcul est refait à chaque exécution à partir des réponses mémorisées : changer de modèle ou de pondérations est immédiat.
inp: UserInput = st.session_state["inp"]
inp.weights = weights
res = engine.recommend(inp)
if not st.session_state.get("logged") and len(res.table):
    log_response(inp, res, engine.backend.label)
    st.session_state["logged"] = True

# ============================================================================================ résultats
st.header("2 · Résultats")
if not res.present:
    st.warning("Aucune préférence exploitable : renseignez au moins une description, des cases à cocher, un repas, une échelle de Likert ou une évaluation de style.")
    st.stop()
for w in res.warnings:
    st.warning(w)
if res.table.empty:
    st.error(f"Aucun vin ne satisfait vos contraintes ({res.n_candidates}/{res.n_total} références après filtres). Élargissez le budget, les catégories ou les pays.")
    st.stop()

tab_rank, tab_viz, tab_profile, tab_method = st.tabs(["🏆 Recommandations", "📈 Visualisations", "🧬 Profil & données", "📚 Méthodologie"])
top = res.table.head(top_n)
wmap = engine.wines.set_index("WineID")

with tab_rank:
    st.info(res.summary)
    st.caption(f"{res.n_candidates}/{res.n_total} références après filtres · composantes utilisées : "
               + " · ".join(f"{COMPONENT_LABELS[c]} ({res.weights[c]:.0%})" for c in res.present))

    cards = st.columns(min(3, len(top)))
    for col, (_, r) in zip(cards, top.head(3).iterrows()):
        with col, st.container(border=True):
            st.markdown(f"### {'🥇🥈🥉'[int(r['Rank']) - 1]} {r['Name']}")
            st.caption(f"{r['Category']} · {r['Region']} · {r['Grape']} · ~{r['PriceEUR']:.0f} €")
            st.progress(float(r["Score"]), text=f"Affinité {r['Score']:.0%} — {r['Level']}")
            for line in engine.explain(res, int(r["idx"]))[:3]:
                st.markdown(f"- {line}")

    st.subheader("Classement complet")
    show = pd.DataFrame({
        "Rang": top["Rank"], "Vin": top["Name"], "Catégorie": top["Category"], "Région": top["Region"], "Prix indicatif (€)": top["PriceEUR"],
        "Score /100": (top["Score"] * 100).round(0), "Niveau": top["Level"],
        **{COMPONENT_LABELS[c]: (top[f"s_{c}"] * 100).round(0) for c in res.present},
        **({"Pénalité (à éviter)": (top["Penalty"] * 100).round(0)} if res.avoided else {}),
    })
    st.dataframe(show, hide_index=True, column_config={"Score /100": st.column_config.ProgressColumn("Score /100", min_value=0, max_value=100, format="%d")}, **STRETCH)
    st.download_button("⬇️ Télécharger le classement (CSV)", show.to_csv(index=False).encode("utf-8"), "recommandations.csv", "text/csv")

    st.subheader("Pourquoi ces vins ?")
    for _, r in top.iterrows():
        wrow = wmap.loc[r["WineID"]]
        with st.expander(f"#{int(r['Rank'])} · {r['Name']} — {r['Score']:.0%}"):
            st.markdown(f"**Description** : {wrow['Description']}")
            st.markdown(f"**Profil sensoriel** : {wrow['SensoryProfile']}")
            st.markdown(f"**Accords mets-vins** : {wrow['FoodPairing']}")
            st.markdown(f"**Mots-clés** : `{'` `'.join(k.strip() for k in wrow['Keywords'].split(','))}`")
            st.markdown("**Détail du score**")
            for line in engine.explain(res, int(r["idx"])):
                st.markdown(f"- {line}")

with tab_viz:
    names = ref.block_names
    st.subheader("Radar des blocs de goût")
    options = list(top["WineID"])
    pick = st.multiselect("Vins à comparer", options, default=options[:3], format_func=lambda i: wmap.loc[i, "Name"])
    fig = go.Figure()
    theta = names + names[:1]
    if res.block_weights is not None:
        rr = res.block_weights / res.block_weights.max()
        fig.add_trace(go.Scatterpolar(r=list(rr) + [rr[0]], theta=theta, name="Votre profil (importance W)", fill="toself", opacity=0.3,
                                      line=dict(dash="dash", color="#222")))
    for wid in pick:
        i = int(engine.wines.index[engine.wines["WineID"] == wid][0])
        s = engine.S_block[i]
        fig.add_trace(go.Scatterpolar(r=list(s) + [s[0]], theta=theta, name=wmap.loc[wid, "Name"], mode="lines+markers"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], tickformat=".0%")), height=480, margin=dict(t=30, b=30), legend=dict(orientation="h", y=-0.1))
    st.plotly_chart(fig, **STRETCH)
    st.caption("Chaque axe = un bloc de goût. Vin : couverture Sᵦ (0-100 % relative au catalogue). Profil : importance Wᵦ déduite de votre texte et de vos cases cochées.")

    st.subheader("Décomposition du score")
    comp_cols = [f"contrib_{c}" for c in res.present] + (["contrib_penalty"] if res.avoided else [])
    long = top.melt(id_vars=["Name"], value_vars=comp_cols, var_name="k", value_name="Contribution")
    long["Composante"] = long["k"].map({**{f"contrib_{c}": COMPONENT_LABELS[c] for c in DEFAULT_WEIGHTS}, "contrib_penalty": "Pénalité (notes à éviter)"})
    bar = px.bar(long, y="Name", x="Contribution", color="Composante", orientation="h", barmode="relative",
                 category_orders={"Name": list(top["Name"])[::-1]}, color_discrete_sequence=px.colors.qualitative.Safe)
    bar.update_layout(height=60 + 55 * len(top), xaxis_tickformat=".0%", yaxis_title=None, legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(bar, **STRETCH)

    st.subheader("Carte de similarité vins × blocs de goût")
    rows, labels = [], []
    if res.block_weights is not None:
        rows.append(res.block_weights / res.block_weights.max())
        labels.append("★ Votre profil")
    for _, r in top.head(10).iterrows():
        rows.append(engine.S_block[int(r["idx"])])
        labels.append(r["Name"])
    heat = px.imshow(np.array(rows), x=names, y=labels, color_continuous_scale=WINE_SCALE, zmin=0, zmax=1, text_auto=".0%", aspect="auto")
    heat.update_layout(height=80 + 45 * len(rows), coloraxis_colorbar=dict(tickformat=".0%"), xaxis=dict(side="top"))
    st.plotly_chart(heat, **STRETCH)

    st.subheader("Carte sémantique du catalogue (projection PCA des embeddings)")
    color_by = st.radio("Colorer par", ["Catégorie", "Cluster sémantique (K-means)"], horizontal=True)
    mf, pos = engine.map_frame(res.user_vec)
    col = "Category" if color_by == "Catégorie" else "Cluster"
    mp = px.scatter(mf, x="x", y="y", color=col, hover_name="Name", hover_data={"Region": True, "PriceEUR": True, "x": False, "y": False},
                    color_discrete_map=CAT_COLORS if col == "Category" else None)
    mp.update_traces(marker=dict(size=10, opacity=0.8))
    t5 = mf[mf["WineID"].isin(res.table.head(5)["WineID"])]
    mp.add_trace(go.Scatter(x=t5["x"], y=t5["y"], mode="markers+text", text=t5["Name"], textposition="top center", name="Top 5",
                            marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(width=2, color="black")), hoverinfo="skip"))
    if pos:
        mp.add_trace(go.Scatter(x=[pos[0]], y=[pos[1]], mode="markers+text", text=["Vous"], textposition="bottom center", name="Votre requête",
                                marker=dict(symbol="star", size=22, color="gold", line=dict(width=2, color="black"))))
    ev = engine.pca.explained_variance_ratio_
    mp.update_layout(height=560, xaxis_title=f"Axe 1 ({ev[0]:.0%} de variance)", yaxis_title=f"Axe 2 ({ev[1]:.0%} de variance)", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(mp, **STRETCH)
    st.caption("Plus deux points sont proches, plus leurs profils sensoriels sont sémantiquement proches. L'étoile est votre description projetée dans le même espace.")

with tab_profile:
    st.subheader("Votre profil synthétisé")
    st.info(res.summary)
    if res.block_weights is not None:
        wdf = pd.DataFrame({"Bloc de goût": names, "Importance Wᵦ": res.block_weights / res.block_weights.max()}).sort_values("Importance Wᵦ")
        wf = px.bar(wdf, x="Importance Wᵦ", y="Bloc de goût", orientation="h", color="Importance Wᵦ", color_continuous_scale=WINE_SCALE, range_color=(0, 1))
        wf.update_layout(height=360, coloraxis_showscale=False, xaxis_tickformat=".0%", yaxis_title=None)
        st.plotly_chart(wf, **STRETCH)
    st.markdown("**Textes analysés par le modèle**")
    st.code(f"Goût / occasion : {res.taste_text or '—'}\nRepas           : {res.food_text or '—'}", language=None)
    st.markdown("**Réponses brutes (JSON structuré)**")
    st.json({k: v for k, v in inp.to_record().items() if v not in ("", [], {}, None)}, expanded=False)
    st.download_button("⬇️ Télécharger mes réponses (JSON)", json.dumps(inp.to_record(), ensure_ascii=False, indent=2).encode("utf-8"), "mes_reponses.json", "application/json")
    st.subheader("Jeu de données de référence")
    st.dataframe(engine.wines.drop(columns=["Country"]), hide_index=True, **STRETCH)
    st.subheader("Blocs de goût (référentiel)")
    st.dataframe(pd.DataFrame({"Bloc": names, "Phrases de référence": [" | ".join(b["phrases"]) for b in ref.blocks]}), hide_index=True, **STRETCH)

with tab_method:
    st.markdown(f"""
### Pipeline NLP
1. **Encodage** — chaque vin est représenté par trois embeddings : *description* (+ catégorie), *profil sensoriel & mots-clés*, *accords mets-vins*.
   Profil global = {PROFILE_W_DESC:.0%} description + {1 - PROFILE_W_DESC:.0%} profil sensoriel. Les réponses libres de l'utilisateur sont encodées par le même modèle.
2. **Similarité** — cosinus entre les embeddings (vecteurs normalisés L2).
3. **Normalisation** — chaque similarité est ramenée sur [0, 1] par min-max sur le catalogue (score *relatif* aux {engine.n} références).
4. **Agrégation pondérée**, puis **pénalité** pour les notes à éviter, puis **classement** après application des filtres (budget, catégorie, pays).
""")
    st.latex(r"\text{Couverture}(v)=\frac{\sum_{b=1}^{9} W_b\,S_b(v)}{\sum_{b=1}^{9} W_b}")
    st.markdown(f"où **S_b(v)** = moyenne des {TOPK_PHRASES} phrases du bloc *b* les plus proches du vin *v* (cosinus, normalisé sur le catalogue) et **W_b** = importance du bloc pour l'utilisateur "
                f"(similarité de son texte aux phrases du bloc, mêlée à ses cases cochées à {1 - BLOCK_MIX:.0%} ; plancher {BLOCK_FLOOR}).")
    st.latex(r"\text{Affinité}(v)=\frac{\sum_{c}\omega_c\,s_c(v)}{\sum_{c}\omega_c}\qquad \text{Score}(v)=\max\bigl(0,\ \text{Affinité}(v)-\gamma\cdot\text{Pénalité}(v)\bigr),\ \gamma=" + f"{AVOID_GAMMA}")
    st.markdown("Composantes *c* : " + " ; ".join(f"**{COMPONENT_LABELS[c]}** (ω par défaut {DEFAULT_WEIGHTS[c]})" for c in DEFAULT_WEIGHTS) + ".")
    st.markdown("Niveaux (cf. étape 5 du sujet de référence) : " + " · ".join(f"≥ {t:.2f} → {n}" if t else f"sinon → {n}" for t, n in LEVELS) + ".")
    st.markdown("""
### Structure (échelles de Likert)
Aucune note de corps / acidité / douceur / tanins n'est saisie à la main : la position d'un vin sur chaque axe vient de sa proximité sémantique à deux jeux de phrases-pôles (« vin très léger… » vs « vin puissant et corsé… »).
Le Likert de l'utilisateur est converti en cible t = (note − 1)/4 ; la composante vaut 1 − |t − position| moyennée sur les axes renseignés.

### Limites assumées
- Jeu de données **fictif et pédagogique** (60 profils types d'appellations, prix indicatifs) ; il ne remplace ni un guide ni un caviste.
- Scores **relatifs au catalogue** : 100 % ne signifie pas « vin parfait », mais « le plus proche parmi les 60 ».
- Le backend `spacy` (vecteurs de mots moyennés) ignore le contexte : il sert de comparaison avec SBERT, pas de moteur de production.
""")

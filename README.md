# Recommandeur sémantique de vins - Thème 5

Projet *Analyse sémantique et systèmes intelligents de recommandation* · ESME ING5 A3MSI 2026-27.
Prototype **pédagogique** (public adulte) : il associe des descriptions en langage naturel (goûts, occasion, repas) à des profils sensoriels de vins, par similarité sémantique d'embeddings.
Aucun achat, aucune incitation à consommer, aucune personnalisation fondée sur des données de santé. *L'abus d'alcool est dangereux pour la santé, à consommer avec modération.*

## Lancer

```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Puis ouvrir `notebook_recommandation_vins.ipynb` (VS Code ou Jupyter, kernel du `.venv`) et lancer « Run All ».

Au premier lancement, SBERT télécharge `paraphrase-multilingual-MiniLM-L12-v2` (~0,5 Go). Hors connexion, ou sans `sentence-transformers`, le notebook bascule seul sur le backend LSA (hors-ligne) et l'indique dans la section 1. Le modèle SBERT se change avec la variable d'environnement `WINE_SBERT_MODEL`.

## Correspondance avec le sujet

| Exigence du sujet | Où / comment |
|---|---|
| Jeu de données ≥ 50 références, structure `WineID \| Category \| Region \| Description \| SensoryProfile \| FoodPairing \| Keywords` | `data/wines.csv` : 60 vins, mêmes colonnes + `Name`, `Grape`, `PriceEUR` (nécessaire au filtre budget) |
| Questionnaire : Likert, questions ouvertes, guidées, choix multiples, cases à cocher | Notebook §7 (objet `UserInput`) : 5 échelles de Likert, 2 questions ouvertes (goûts, repas), 3 questions guidées, choix unique (occasion, plat) et multiple (notes à éviter, catégories, pays), 9 cases à cocher, 6 cartes de style notées de 1 à 5 |
| Informations attendues : texte libre, caractéristiques aromatiques, contexte mets-vins, évaluations, budget/contraintes | respectivement : zone de texte + occasion · cases à cocher · type de plat + texte · Likert + cartes de style · budget, catégorie, pays |
| Embeddings contextuels (SBERT) + similarité cosinus | `embeddings.py::SBERTBackend` (backend principal) |
| Embeddings de mots (Word2Vec / fastText) | `SpacyBackend` : moyenne de vecteurs fastText/word2vec de `fr_core_news_md` (baseline de comparaison) |
| Clustering sémantique | K-means (k = 6) sur les embeddings des vins, affiché sur la carte sémantique |
| Score d'affinité avec mécanisme de scoring défini et justifié | `wine_engine.py` + section *Méthodologie* ci-dessous et notebook §9 (calcul pas à pas) |
| Score de couverture agrégeant plusieurs blocs | 9 blocs de goût, formule `Σ Wb·Sb / Σ Wb` (équivalent de l'exemple « blocs de compétences ») |
| Profil utilisateur synthétisé puis recommandations | « Profil détecté » + classement + explications par vin (notebook §8 et §10) |
| Visualisations | radar des blocs, barres empilées (décomposition du score), carte de similarité vins × blocs, carte PCA du catalogue |
| Stockage structuré des réponses | notebook §14 : enregistrement JSON de chaque réponse (`responses/notebook_responses.jsonl`, option `SAVE_RESPONSE`) |
| Choix documentés et justifiés | ce README (section *Choix à justifier*) |

## Structure du projet

```
notebook_recommandation_vins.ipynb   le livrable : données, embeddings, scoring pas à pas, visualisations, évaluation
wine_engine.py     moteur : embeddings du catalogue, blocs, axes, scoring, filtres, explications
embeddings.py      backends interchangeables : sbert · spacy · lsa
data/wines.csv     60 vins                      data/reference.json   blocs, axes, cartes de style, occasions, plats, questions guidées
evaluate.py        19 requêtes de test et métriques (utilisé par le notebook)   docs/evaluation_results.md   dernier résultat
```

Tout le référentiel sémantique est dans `data/` : on ajoute un vin (une ligne de CSV) ou une phrase de bloc sans toucher au code.

## Méthodologie

**1. Encodage du catalogue.** Chaque vin a trois embeddings : *description* (préfixée par la catégorie), *profil sensoriel + mots-clés*, *accords mets-vins*. Le profil global est `0,4·description + 0,6·sensoriel`, renormalisé. Les champs sont encodés séparément pour rester sous la fenêtre de 128 tokens de SBERT et pour comparer chaque champ à la bonne entrée utilisateur.

**2. Référentiel de blocs de goût** (équivalent des blocs de compétences). 9 blocs (fruits rouges/noirs, agrumes/fruits blancs, floral/herbacé, épices/garrigue, boisé/torréfié, minéral/salin, fraîcheur, rondeur/douceur, structure/puissance), chacun décrit par 5 courtes phrases.

**3. Score d'un vin pour un bloc** : `Sb(v)` = moyenne des 2 phrases du bloc les plus proches du vin (cosinus), puis min-max sur le catalogue par bloc.

**4. Importance des blocs pour l'utilisateur** : `Wb` = similarité de son texte libre/occasion aux phrases du bloc (min-max puis carré, pour ne garder que les blocs saillants), mélangée à 50 % avec ses cases cochées ; plancher à 0,05.

**5. Composantes du score** (chacune dans [0, 1]) :

| Composante | Calcul | Poids ω par défaut |
|---|---|---|
| Affinité sémantique | cos(texte goût/occasion/familles cochées, profil du vin), min-max catalogue | 0,30 |
| Couverture des blocs | `Σ Wb·Sb / Σ Wb` | 0,25 |
| Accord mets-vins | cos(texte du repas, champ `FoodPairing`), min-max catalogue | 0,20 |
| Structure (Likert) | `1 − \|cible − position\|` moyennée sur les axes renseignés ; cible = (note−1)/4 | 0,15 |
| Préférences de style | Σ (note−3)·cos(vin, carte) / Σ \|note−3\|, min-max | 0,10 |

`Affinité = Σ ωc·sc / Σ ωc` sur les **seules composantes renseignées** (poids renormalisés), puis `Score = max(0, Affinité − 0,30 × Pénalité)`, où la pénalité est la couverture moyenne des blocs que l'utilisateur veut éviter. Budget, catégorie et pays sont des **filtres durs** appliqués après calibration sur tout le catalogue. Les curseurs de la barre latérale modifient les ω en direct.

**6. Axes de structure sans note manuelle.** Corps, acidité, douceur, tanins, intensité : la position d'un vin sur un axe vient de sa proximité à deux jeux de phrases-pôles (« vin très léger… » / « vin puissant et corsé… »), mise à l'échelle [P5, P95] sur le catalogue.

**7. Niveaux** (cf. étape 5 du sujet) : ≥ 0,70 *très bonne affinité* · 0,50–0,70 *bonne* · sinon *partielle*.

## Choix à justifier (à reprendre dans le rapport)

- **SBERT multilingue** : les descriptions sont en français et très contextuelles (« tanins soyeux » ≠ « tanins rugueux ») ; un modèle de phrases les compare sans exiger de mots communs. Dans nos essais, les vecteurs de mots moyennés ont par exemple remonté « Côte-Rôtie » pour une requête sur la « côte de bœuf ».
- **Trois backends derrière une même interface** : permet de comparer techniques (contextuel, mots, latent) et garantit que l'application tourne sans téléchargement.
- **Min-max sur le catalogue** : la plage des cosinus dépend du modèle (SBERT ≠ LSA ≠ spaCy) ; la normalisation rend les composantes additionnables quel que soit le modèle. Contrepartie : scores *relatifs* à la requête et au catalogue (100 % = le plus proche des 60), donc les seuils 0,5 / 0,7 des niveaux ne sont pas des seuils absolus et ne se comparent pas d'un backend à l'autre.
- **Moyenne des 2 meilleures phrases par bloc** : une phrase seule est trop sensible au bruit, la moyenne de toutes dilue (un vin n'exprime en général qu'une partie d'un bloc).
- **Profil 0,4 / 0,6** : le profil sensoriel porte l'information aromatique la plus dense ; la description apporte le style et l'occasion.
- **Pondérations par défaut** : le texte libre est le signal le plus riche (0,30) ; couverture et accords suivent ; Likert et notes de style, plus grossiers, pèsent moins. Elles sont ajustables et ne sont pas apprises (pas de données d'usage).
- **Pénalité plutôt qu'exclusion** pour les notes à éviter : une préférence négative est rarement absolue.
- **Cases cochées *et* texte pour W** : donne un signal explicite robuste si le texte est court, sans écraser la nuance du texte.

## Évaluation (`python evaluate.py`)

19 requêtes en français (13 directes, 6 paraphrases volontairement éloignées du vocabulaire du jeu de données), jeux de vins pertinents annotés à la main avant calcul, un seul annotateur : c'est un **test de cohérence, pas un benchmark**. P@5 / Hit@1 / MRR moyens, mode « texte » :

| Backend | Directes P@5 / Hit@1 / MRR | Paraphrases P@5 / Hit@1 / MRR |
|---|---|---|
| Mots-clés (TF-IDF brut, baseline) | 0,57 / 0,85 / 0,90 | 0,33 / 0,50 / 0,66 |
| spaCy (vecteurs de mots moyennés) | 0,35 / 0,69 / 0,79 | 0,43 / 0,50 / 0,66 |
| LSA (TF-IDF + SVD, 40 dim.) | 0,60 / 0,85 / 0,89 | 0,50 / 0,67 / 0,72 |
| **SBERT** | **à mesurer sur votre machine** | **à mesurer sur votre machine** |

À lire avec prudence : sur les requêtes directes, LSA n'est que légèrement au-dessus des mots-clés bruts (le jeu de données et les requêtes partagent beaucoup de vocabulaire) ; l'écart se voit sur les paraphrases. Une requête dont aucun mot n'est connu de LSA ou de spaCy ne renvoie rien (le moteur le signale) ; SBERT n'a pas cette limite. Cohérence des axes de structure (AUC, 1 = parfait) : LSA 1,00 / 1,00 / 0,99 / 0,94 et spaCy 1,00 / 0,73 / 0,87 / 0,90 pour corps / acidité / douceur / tanins.

## Vérification

- Le notebook recalcule à la main la formule de couverture et le score final du vin n°1 et les compare au moteur (`assert`, sections 9.1 et 9.2).
- **Le backend SBERT n'a pas pu être exécuté dans l'environnement de développement** (le téléchargement du modèle depuis huggingface.co y est bloqué) : son code est standard (`SentenceTransformer(...).encode(..., normalize_embeddings=True)`) mais son comportement n'a pas été mesuré. Relancez le notebook avec `sentence-transformers` installé : SBERT est utilisé automatiquement et ajouté au tableau de la section 13.

## Limites et éthique

- Jeu de données **fictif et pédagogique** : 60 profils types d'appellations rédigés pour le projet, prix indicatifs ; ce n'est ni un guide ni un conseil d'achat.
- Pas d'achat, pas de lien marchand, pas de donnée de santé demandée ; public adulte uniquement. Les réponses enregistrées par le notebook (option `SAVE_RESPONSE`) restent en local dans `responses/` : ne pas y saisir d'informations personnelles.
- Scores relatifs au catalogue ; poids non appris.

## Pistes (bonus facultatif du sujet)

Chatbot « sommelier » expliquant un classement (LLM + RAG sur `wines.csv`), reformulation des réponses libres avant encodage, apprentissage des poids à partir des retours utilisateurs, enrichissement du catalogue (≥ 200 vins) pour des clusters plus stables.

# Résultats d'évaluation

19 requêtes-tests en français (13 directes + 6 paraphrases), jeux de vins pertinents annotés à la main (annotateur unique) : test de cohérence, pas benchmark.

## Qualité du classement

| Backend                                    | Mode        | Groupe      |   P@5 |   Hit@1 |   MRR |
|:-------------------------------------------|:------------|:------------|------:|--------:|------:|
| Mots-clés (TF-IDF brut, baseline)          | —           | directes    | 0.569 |   0.846 | 0.904 |
| Mots-clés (TF-IDF brut, baseline)          | —           | paraphrases | 0.333 |   0.5   | 0.658 |
| spaCy · fr_core_news_md (vecteurs de mots) | texte       | directes    | 0.354 |   0.692 | 0.785 |
| spaCy · fr_core_news_md (vecteurs de mots) | texte+repas | directes    | 0.4   |   0.615 | 0.767 |
| spaCy · fr_core_news_md (vecteurs de mots) | texte       | paraphrases | 0.433 |   0.5   | 0.658 |
| spaCy · fr_core_news_md (vecteurs de mots) | texte+repas | paraphrases | 0.5   |   0.667 | 0.764 |
| LSA · TF-IDF + SVD (hors-ligne)            | texte       | directes    | 0.6   |   0.846 | 0.891 |
| LSA · TF-IDF + SVD (hors-ligne)            | texte+repas | directes    | 0.615 |   0.846 | 0.904 |
| LSA · TF-IDF + SVD (hors-ligne)            | texte       | paraphrases | 0.5   |   0.667 | 0.722 |
| LSA · TF-IDF + SVD (hors-ligne)            | texte+repas | paraphrases | 0.567 |   0.5   | 0.667 |

## Cohérence des axes de structure (AUC)

| Backend                                    |   body |   acidity |   sweetness |   tannin |
|:-------------------------------------------|-------:|----------:|------------:|---------:|
| spaCy · fr_core_news_md (vecteurs de mots) |      1 |      0.73 |        0.87 |     0.9  |
| LSA · TF-IDF + SVD (hors-ligne)            |      1 |      1    |        0.99 |     0.94 |

## Détail par requête

| Backend                                    | Mode        | Groupe      | Requête                                                                             |   P@5 |   Hit@1 |   MRR |
|:-------------------------------------------|:------------|:------------|:------------------------------------------------------------------------------------|------:|--------:|------:|
| Mots-clés (TF-IDF brut, baseline)          | —           | directes    | un vin blanc vif et minéral pour accompagner des huîtres                            |   0.6 |       1 |  1    |
| Mots-clés (TF-IDF brut, baseline)          | —           | directes    | un rouge puissant et tannique pour une côte de bœuf grillée                         |   1   |       1 |  1    |
| Mots-clés (TF-IDF brut, baseline)          | —           | directes    | un vin doux pour accompagner un foie gras                                           |   0.6 |       1 |  1    |
| Mots-clés (TF-IDF brut, baseline)          | —           | directes    | un rosé frais et fruité pour un apéritif d'été                                      |   0.6 |       1 |  1    |
| Mots-clés (TF-IDF brut, baseline)          | —           | directes    | un vin léger et fruité, facile à boire entre amis, sur la fraise et la cerise       |   0.2 |       0 |  0.25 |
| Mots-clés (TF-IDF brut, baseline)          | —           | directes    | un vin aromatique sur le litchi et la rose pour un plat asiatique épicé             |   0.2 |       1 |  1    |
| Mots-clés (TF-IDF brut, baseline)          | —           | directes    | des bulles fines pour célébrer un repas de fête                                     |   0.8 |       1 |  1    |
| Mots-clés (TF-IDF brut, baseline)          | —           | directes    | un vin boisé, vanillé et rond avec du beurre                                        |   0.8 |       1 |  1    |
| Mots-clés (TF-IDF brut, baseline)          | —           | directes    | un rouge poivré avec des notes de garrigue pour un barbecue entre amis              |   0.6 |       1 |  1    |
| Mots-clés (TF-IDF brut, baseline)          | —           | directes    | un blanc citronné et herbacé pour une salade et du fromage de chèvre                |   0.2 |       0 |  0.5  |
| Mots-clés (TF-IDF brut, baseline)          | —           | directes    | un vin doux et léger pour accompagner une tarte aux fruits                          |   0.4 |       1 |  1    |
| Mots-clés (TF-IDF brut, baseline)          | —           | directes    | un vin de garde très complexe, avec du cassis et du cèdre, pour une grande occasion |   0.6 |       1 |  1    |
| Mots-clés (TF-IDF brut, baseline)          | —           | directes    | un vin salin et iodé pour des moules et des crevettes                               |   0.8 |       1 |  1    |
| Mots-clés (TF-IDF brut, baseline)          | —           | paraphrases | un vin qui sent la mer, parfait avec des coquillages                                |   0.4 |       1 |  1    |
| Mots-clés (TF-IDF brut, baseline)          | —           | paraphrases | un rouge costaud qui laisse la bouche sèche, pour du gibier                         |   0.4 |       0 |  0.5  |
| Mots-clés (TF-IDF brut, baseline)          | —           | paraphrases | un vin tout sucré avec un goût de confiture d'abricot, pour le dessert              |   0.4 |       0 |  0.25 |
| Mots-clés (TF-IDF brut, baseline)          | —           | paraphrases | une boisson pétillante pour trinquer avec des invités                               |   0.2 |       0 |  0.2  |
| Mots-clés (TF-IDF brut, baseline)          | —           | paraphrases | un vin tendre et facile pour quelqu'un qui découvre le vin                          |   0.2 |       1 |  1    |
| Mots-clés (TF-IDF brut, baseline)          | —           | paraphrases | un vin qui sent les fleurs et les fruits exotiques, très onctueux                   |   0.4 |       1 |  1    |
| spaCy · fr_core_news_md (vecteurs de mots) | texte       | directes    | un vin blanc vif et minéral pour accompagner des huîtres                            |   0.4 |       0 |  0.5  |
| spaCy · fr_core_news_md (vecteurs de mots) | texte+repas | directes    | un vin blanc vif et minéral pour accompagner des huîtres                            |   0.2 |       0 |  0.33 |
| spaCy · fr_core_news_md (vecteurs de mots) | texte       | directes    | un rouge puissant et tannique pour une côte de bœuf grillée                         |   0.2 |       1 |  1    |
| spaCy · fr_core_news_md (vecteurs de mots) | texte+repas | directes    | un rouge puissant et tannique pour une côte de bœuf grillée                         |   0.2 |       0 |  0.5  |
| spaCy · fr_core_news_md (vecteurs de mots) | texte       | directes    | un vin doux pour accompagner un foie gras                                           |   0.6 |       1 |  1    |
| spaCy · fr_core_news_md (vecteurs de mots) | texte+repas | directes    | un vin doux pour accompagner un foie gras                                           |   0.8 |       1 |  1    |
| spaCy · fr_core_news_md (vecteurs de mots) | texte       | directes    | un rosé frais et fruité pour un apéritif d'été                                      |   0.4 |       1 |  1    |
| spaCy · fr_core_news_md (vecteurs de mots) | texte+repas | directes    | un rosé frais et fruité pour un apéritif d'été                                      |   0.6 |       1 |  1    |
| spaCy · fr_core_news_md (vecteurs de mots) | texte       | directes    | un vin léger et fruité, facile à boire entre amis, sur la fraise et la cerise       |   0.2 |       1 |  1    |
| spaCy · fr_core_news_md (vecteurs de mots) | texte+repas | directes    | un vin léger et fruité, facile à boire entre amis, sur la fraise et la cerise       |   0.2 |       0 |  0.5  |
| spaCy · fr_core_news_md (vecteurs de mots) | texte       | directes    | un vin aromatique sur le litchi et la rose pour un plat asiatique épicé             |   0.2 |       1 |  1    |
| spaCy · fr_core_news_md (vecteurs de mots) | texte+repas | directes    | un vin aromatique sur le litchi et la rose pour un plat asiatique épicé             |   0.2 |       0 |  0.5  |
| spaCy · fr_core_news_md (vecteurs de mots) | texte       | directes    | des bulles fines pour célébrer un repas de fête                                     |   0.4 |       1 |  1    |
| spaCy · fr_core_news_md (vecteurs de mots) | texte+repas | directes    | des bulles fines pour célébrer un repas de fête                                     |   0.4 |       1 |  1    |
| spaCy · fr_core_news_md (vecteurs de mots) | texte       | directes    | un vin boisé, vanillé et rond avec du beurre                                        |   0.6 |       1 |  1    |
| spaCy · fr_core_news_md (vecteurs de mots) | texte+repas | directes    | un vin boisé, vanillé et rond avec du beurre                                        |   0.6 |       1 |  1    |
| spaCy · fr_core_news_md (vecteurs de mots) | texte       | directes    | un rouge poivré avec des notes de garrigue pour un barbecue entre amis              |   0   |       0 |  0.14 |
| spaCy · fr_core_news_md (vecteurs de mots) | texte+repas | directes    | un rouge poivré avec des notes de garrigue pour un barbecue entre amis              |   0.2 |       1 |  1    |
| spaCy · fr_core_news_md (vecteurs de mots) | texte       | directes    | un blanc citronné et herbacé pour une salade et du fromage de chèvre                |   0   |       0 |  0.06 |
| spaCy · fr_core_news_md (vecteurs de mots) | texte+repas | directes    | un blanc citronné et herbacé pour une salade et du fromage de chèvre                |   0   |       0 |  0.14 |
| spaCy · fr_core_news_md (vecteurs de mots) | texte       | directes    | un vin doux et léger pour accompagner une tarte aux fruits                          |   0.6 |       1 |  1    |
| spaCy · fr_core_news_md (vecteurs de mots) | texte+repas | directes    | un vin doux et léger pour accompagner une tarte aux fruits                          |   0.6 |       1 |  1    |
| spaCy · fr_core_news_md (vecteurs de mots) | texte       | directes    | un vin de garde très complexe, avec du cassis et du cèdre, pour une grande occasion |   0.6 |       1 |  1    |
| spaCy · fr_core_news_md (vecteurs de mots) | texte+repas | directes    | un vin de garde très complexe, avec du cassis et du cèdre, pour une grande occasion |   0.6 |       1 |  1    |
| spaCy · fr_core_news_md (vecteurs de mots) | texte       | directes    | un vin salin et iodé pour des moules et des crevettes                               |   0.4 |       0 |  0.5  |
| spaCy · fr_core_news_md (vecteurs de mots) | texte+repas | directes    | un vin salin et iodé pour des moules et des crevettes                               |   0.6 |       1 |  1    |
| spaCy · fr_core_news_md (vecteurs de mots) | texte       | paraphrases | un vin qui sent la mer, parfait avec des coquillages                                |   0.6 |       1 |  1    |
| spaCy · fr_core_news_md (vecteurs de mots) | texte+repas | paraphrases | un vin qui sent la mer, parfait avec des coquillages                                |   0.8 |       1 |  1    |
| spaCy · fr_core_news_md (vecteurs de mots) | texte       | paraphrases | un rouge costaud qui laisse la bouche sèche, pour du gibier                         |   0.4 |       0 |  0.25 |
| spaCy · fr_core_news_md (vecteurs de mots) | texte+repas | paraphrases | un rouge costaud qui laisse la bouche sèche, pour du gibier                         |   0.4 |       0 |  0.33 |
| spaCy · fr_core_news_md (vecteurs de mots) | texte       | paraphrases | un vin tout sucré avec un goût de confiture d'abricot, pour le dessert              |   0.4 |       1 |  1    |
| spaCy · fr_core_news_md (vecteurs de mots) | texte+repas | paraphrases | un vin tout sucré avec un goût de confiture d'abricot, pour le dessert              |   0.4 |       1 |  1    |
| spaCy · fr_core_news_md (vecteurs de mots) | texte       | paraphrases | une boisson pétillante pour trinquer avec des invités                               |   0.2 |       0 |  0.2  |
| spaCy · fr_core_news_md (vecteurs de mots) | texte+repas | paraphrases | une boisson pétillante pour trinquer avec des invités                               |   0.4 |       0 |  0.25 |
| spaCy · fr_core_news_md (vecteurs de mots) | texte       | paraphrases | un vin tendre et facile pour quelqu'un qui découvre le vin                          |   0.6 |       0 |  0.5  |
| spaCy · fr_core_news_md (vecteurs de mots) | texte+repas | paraphrases | un vin tendre et facile pour quelqu'un qui découvre le vin                          |   0.8 |       1 |  1    |
| spaCy · fr_core_news_md (vecteurs de mots) | texte       | paraphrases | un vin qui sent les fleurs et les fruits exotiques, très onctueux                   |   0.4 |       1 |  1    |
| spaCy · fr_core_news_md (vecteurs de mots) | texte+repas | paraphrases | un vin qui sent les fleurs et les fruits exotiques, très onctueux                   |   0.2 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte       | directes    | un vin blanc vif et minéral pour accompagner des huîtres                            |   0.6 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte+repas | directes    | un vin blanc vif et minéral pour accompagner des huîtres                            |   1   |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte       | directes    | un rouge puissant et tannique pour une côte de bœuf grillée                         |   0.6 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte+repas | directes    | un rouge puissant et tannique pour une côte de bœuf grillée                         |   0.8 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte       | directes    | un vin doux pour accompagner un foie gras                                           |   0.4 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte+repas | directes    | un vin doux pour accompagner un foie gras                                           |   0.6 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte       | directes    | un rosé frais et fruité pour un apéritif d'été                                      |   0.6 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte+repas | directes    | un rosé frais et fruité pour un apéritif d'été                                      |   0.4 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte       | directes    | un vin léger et fruité, facile à boire entre amis, sur la fraise et la cerise       |   0.6 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte+repas | directes    | un vin léger et fruité, facile à boire entre amis, sur la fraise et la cerise       |   0.6 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte       | directes    | un vin aromatique sur le litchi et la rose pour un plat asiatique épicé             |   0.2 |       0 |  0.25 |
| LSA · TF-IDF + SVD (hors-ligne)            | texte+repas | directes    | un vin aromatique sur le litchi et la rose pour un plat asiatique épicé             |   0.2 |       0 |  0.25 |
| LSA · TF-IDF + SVD (hors-ligne)            | texte       | directes    | des bulles fines pour célébrer un repas de fête                                     |   1   |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte+repas | directes    | des bulles fines pour célébrer un repas de fête                                     |   0.8 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte       | directes    | un vin boisé, vanillé et rond avec du beurre                                        |   0.8 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte+repas | directes    | un vin boisé, vanillé et rond avec du beurre                                        |   0.6 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte       | directes    | un rouge poivré avec des notes de garrigue pour un barbecue entre amis              |   0.8 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte+repas | directes    | un rouge poivré avec des notes de garrigue pour un barbecue entre amis              |   0.6 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte       | directes    | un blanc citronné et herbacé pour une salade et du fromage de chèvre                |   0.2 |       0 |  0.33 |
| LSA · TF-IDF + SVD (hors-ligne)            | texte+repas | directes    | un blanc citronné et herbacé pour une salade et du fromage de chèvre                |   0.4 |       0 |  0.5  |
| LSA · TF-IDF + SVD (hors-ligne)            | texte       | directes    | un vin doux et léger pour accompagner une tarte aux fruits                          |   0.6 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte+repas | directes    | un vin doux et léger pour accompagner une tarte aux fruits                          |   0.6 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte       | directes    | un vin de garde très complexe, avec du cassis et du cèdre, pour une grande occasion |   0.6 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte+repas | directes    | un vin de garde très complexe, avec du cassis et du cèdre, pour une grande occasion |   0.6 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte       | directes    | un vin salin et iodé pour des moules et des crevettes                               |   0.8 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte+repas | directes    | un vin salin et iodé pour des moules et des crevettes                               |   0.8 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte       | paraphrases | un vin qui sent la mer, parfait avec des coquillages                                |   0.6 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte+repas | paraphrases | un vin qui sent la mer, parfait avec des coquillages                                |   0.8 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte       | paraphrases | un rouge costaud qui laisse la bouche sèche, pour du gibier                         |   0.6 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte+repas | paraphrases | un rouge costaud qui laisse la bouche sèche, pour du gibier                         |   1   |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte       | paraphrases | un vin tout sucré avec un goût de confiture d'abricot, pour le dessert              |   0.6 |       0 |  0.33 |
| LSA · TF-IDF + SVD (hors-ligne)            | texte+repas | paraphrases | un vin tout sucré avec un goût de confiture d'abricot, pour le dessert              |   0.6 |       0 |  0.5  |
| LSA · TF-IDF + SVD (hors-ligne)            | texte       | paraphrases | une boisson pétillante pour trinquer avec des invités                               |   0   |       0 |  0    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte+repas | paraphrases | une boisson pétillante pour trinquer avec des invités                               |   0   |       0 |  0    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte       | paraphrases | un vin tendre et facile pour quelqu'un qui découvre le vin                          |   0.6 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte+repas | paraphrases | un vin tendre et facile pour quelqu'un qui découvre le vin                          |   0.6 |       0 |  0.5  |
| LSA · TF-IDF + SVD (hors-ligne)            | texte       | paraphrases | un vin qui sent les fleurs et les fruits exotiques, très onctueux                   |   0.6 |       1 |  1    |
| LSA · TF-IDF + SVD (hors-ligne)            | texte+repas | paraphrases | un vin qui sent les fleurs et les fruits exotiques, très onctueux                   |   0.4 |       1 |  1    |

> `sbert` indisponible → ignoré (Backend « sbert » indisponible (ModuleNotFoundError: No module named 'sentence_transformers').; Repli sur le backend hors-ligne LSA.)
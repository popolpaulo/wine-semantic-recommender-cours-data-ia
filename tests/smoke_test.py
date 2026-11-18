"""Test de fumée de l'interface (sans navigateur) : `python tests/smoke_test.py [backend]`."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

backend = sys.argv[1] if len(sys.argv) > 1 else "lsa"
at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
at.run()
assert not at.exception, at.exception
# 1) porte « adultes »
assert any("Cochez la case" in i.value for i in at.info), "la porte d'accès devrait bloquer le questionnaire"
at.checkbox(key="adult_ok").check().run()
assert not at.exception, at.exception
# 2) choix du backend
at.selectbox(key="backend_choice").select(backend).run()
assert not at.exception, at.exception
# 3) remplissage du questionnaire
at.text_area[0].set_value("J'aime les vins frais et légers, sur la fraise et la cerise, pour un pique-nique")
at.checkbox(key="like_B1").check()
at.radio[0].set_value("Pique-nique / été")
at.select_slider(key="lk_body").set_value(2)
at.select_slider(key="lk_acidity").set_value(4)
at.select_slider(key="sc_S4").set_value(5)
at.select_slider(key="sc_S2").set_value(1)
at.text_area[1].set_value("charcuterie et fromages")
btn = [b for b in at.button if "recommandations" in b.label][0]
btn.click().run()
assert not at.exception, at.exception
assert "inp" in at.session_state
tabs = [t.label for t in at.tabs]
print("onglets :", tabs)
assert any("Recommandations" in t for t in tabs)
print("dataframes :", len(at.dataframe), "| metrics/progress OK | exceptions :", len(at.exception))
inp = at.session_state["inp"]
print("réponses mémorisées :", {k: v for k, v in inp.to_record().items() if v not in ("", [], {}, None)})
# 4) changer la pondération → recalcul sans soumission
at.slider(key="w_food").set_value(0.0).run()
assert not at.exception, at.exception
assert any(backend in s.value.lower() or backend == "lsa" and "LSA" in s.value for s in at.success), [s.value for s in at.success]
print("OK — interface testée avec le backend", backend, "|", [s.value for s in at.success][0])

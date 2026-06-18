import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cajun8h.cajun_lexicon import LEXICON, respell, to_js


def test_bayou_boat_and_fishing_terms_are_respelled():
    text = "On prend la pirogue dans le bayou pour aller à la pêche."

    assert respell(text) == "On prend la pi-ro dans le baïou pour aller à la pèch."


def test_water_and_shrimp_terms_include_ascii_variants():
    text = "Les crevettes dans l'eau salee et les crevette dans l'eau douce."

    assert respell(text) == "Les chevrettes dans l'eau salé et les chevrette dans l'eau dous."


def test_cajun_lexicon_js_export_includes_new_entries():
    emitted = to_js()

    assert LEXICON["pirogue"] == "pi-ro"
    assert LEXICON["crevette"] == "chevrette"
    assert '"pirogue": "pi-ro"' in emitted
    assert '"crevette": "chevrette"' in emitted
def check_respells_acadiana_town_names():
    text = "Ville Platte, Gueydan, Duson, Erath, and Pierre Part."

    assert respell(text) == "Ville Plat, Guédan, Duzon, Érat, and Pierre Par."


def check_exports_acadiana_town_names_to_js():
    js = to_js()

    assert '"Ville Platte": "Ville Plat"' in js
    assert '"Mermentau": "Mermento"' in js
    assert '"Meraux": "Méro"' in js


def test_respells_acadiana_town_names():
    check_respells_acadiana_town_names()


def test_exports_acadiana_town_names_to_js():
    check_exports_acadiana_town_names_to_js()


if __name__ == "__main__":
    check_respells_acadiana_town_names()
    check_exports_acadiana_town_names_to_js()

import importlib.util
from pathlib import Path

from scripts.cajun8h import cajun_lexicon
from scripts.cajun8h.cajun_lexicon import LEXICON, respell, to_js


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cajun8h" / "cajun_lexicon.py"


def load_module():
    spec = importlib.util.spec_from_file_location("cajun_lexicon", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_respell_cajun_surnames_uses_acadian_family_name_entries():
    text = "Comeaux, Arceneaux, Broussard, Fontenot, and Bourgeois arrived."

    assert (
        respell(text)
        == "Como, Arsenô, Broussar, Fontenô, and Bourjwa arrived."
    )


def test_respell_cajun_surnames_keeps_ascii_variants_and_capitalization():
    text = "LeBlanc met Theriot, Sonnier, Doucet, Dugas, and Richard."

    assert (
        respell(text)
        == "Le Blan met Tério, Sonyé, Doucé, Duga, and Richar."
    )


def test_cajun_lexicon_js_export_includes_surname_entries():
    js = to_js()

    assert '"Comeaux": "Como"' in js
    assert '"Bourgeois": "Bourjwa"' in js


def test_respell_cajun_culinary_terms():
    text = "Boudin, beignets, courtbouillon, roux, and boulettes."

    assert (
        cajun_lexicon.respell(text)
        == "Bou-dan, bèn-yés, cou-bouyon, rou, and boulets."
    )


def test_respell_cajun_culinary_multiword_terms():
    text = "Serve cochon de lait with tarte a la bouillie and cafe au lait."

    assert (
        cajun_lexicon.respell(text)
        == "Serve cou-shon de lait with tarte à la bou-yee and café ô lait."
    )


def test_cajun_lexicon_js_export_includes_culinary_terms():
    js = cajun_lexicon.to_js()

    assert '"boudin": "bou-dan"' in js
    assert '"courtbouillon": "cou-bouyon"' in js
    assert '"remoulade": "rémoularde"' in js


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


def test_household_lsu_glossary_respellings():
    lexicon = load_module()

    text = "Le capot ciré, le chapelet, la canaille, et le hibou."

    assert lexicon.respell(text) == "Le capo ciré, le chaplé, la canaï, et le ibou."


def test_ascii_variants_and_phrase_respellings():
    lexicon = load_module()

    text = "Carrement, hormis que les grands helas parlent du hamecon."

    assert lexicon.respell(text) == "Carrèman, ormi que les grands éla parlent du am-son."


def test_to_js_includes_new_entries():
    lexicon = load_module()

    js = lexicon.to_js()

    assert '"capot ciré": "capo ciré"' in js
    assert '"chaste-femme": "chasse-femme"' in js


if __name__ == "__main__":
    check_respells_acadiana_town_names()
    check_exports_acadiana_town_names_to_js()

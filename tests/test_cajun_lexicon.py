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

# SPDX-License-Identifier: MIT
from scripts.cajun8h.cajun_lexicon import respell, to_js


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


def test_cajun_lexicon_js_export_includes_new_entries():
    js = to_js()

    assert '"Comeaux": "Como"' in js
    assert '"Bourgeois": "Bourjwa"' in js

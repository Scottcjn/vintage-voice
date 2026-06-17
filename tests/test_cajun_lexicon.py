# SPDX-License-Identifier: MIT
from scripts.cajun8h import cajun_lexicon


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

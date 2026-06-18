import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cajun8h" / "cajun_lexicon.py"


def load_module():
    spec = importlib.util.spec_from_file_location("cajun_lexicon", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

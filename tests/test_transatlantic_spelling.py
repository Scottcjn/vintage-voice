from scripts.transatlantic_spelling import respell_transatlantic


def test_respell_transatlantic_replaces_multi_word_phrases_first():
    text = "My dear, that is quite so."

    assert respell_transatlantic(text) == "mai deah, that is kwaite soh."


def test_respell_transatlantic_is_case_insensitive_and_preserves_punctuation():
    text = "Rather, FATHER! Are you there?"

    assert respell_transatlantic(text) == "rahthuh, fahthuh! ah you theah?"


def test_respell_transatlantic_uses_word_boundaries():
    text = "The cargo car started near the carpet."

    result = respell_transatlantic(text)

    assert "cahgo" not in result
    assert "cahpet" not in result
    assert result == "thee cargo cah started near thee carpet."


def test_respell_transatlantic_handles_contractions_and_broad_a_words():
    text = "I can't ask my mother to dance after class."

    assert (
        respell_transatlantic(text)
        == "ai cahnt ahsk mai muhthuh to dahnce ahftuh clahss."
    )


def test_respell_transatlantic_leaves_unknown_words_unchanged():
    text = "Quantum zucchini remains unaltered."

    assert respell_transatlantic(text) == text

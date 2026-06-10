#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
VintageVoice — Transatlantic Phonetic Re-Spelling

Since F5-TTS doesn't easily learn accent transfer, we force transatlantic
pronunciation by re-spelling words phonetically before passing to the model.

Key transatlantic features:
- Non-rhotic: drop R at end of words or before consonants
- Broad A: dance → dahnce, bath → bahth
- Rounded O: hot → hawt
- Clipped T: better → bettuh
- Theatrical vowel elongation
"""
import re


# Transatlantic respelling dictionary
# Format: lowercase word → transatlantic phonetic spelling
TRANSATLANTIC_RESPELL = {
    # Non-rhotic (drop/soften final R)
    "rather": "rahthuh",
    "father": "fahthuh",
    "mother": "muhthuh",
    "brother": "bruhthuh",
    "other": "uhthuh",
    "another": "uhnuhthuh",
    "over": "oh-vuh",
    "never": "nevvuh",
    "ever": "evvuh",
    "water": "wahtuh",
    "better": "bettuh",
    "butter": "buttuh",
    "after": "ahftuh",
    "matter": "mahttuh",
    "later": "lay-tuh",
    "greater": "gray-tuh",
    "sister": "sistuh",
    "under": "uhn-duh",
    "winter": "win-tuh",
    "summer": "summuh",
    "dinner": "dinnuh",
    "center": "sen-tuh",
    "number": "num-buh",
    "remember": "remembuh",
    "together": "togethuh",
    "whether": "wethuh",
    "weather": "wethuh",
    "corner": "cornuh",
    "paper": "pay-puh",
    "proper": "prop-puh",
    "higher": "high-uh",
    "lower": "low-uh",
    "power": "pow-uh",
    "hour": "ow-uh",
    "dear": "deah",
    "here": "heah",
    "there": "theah",
    "where": "wheah",
    "before": "befoh",
    "more": "moh",
    "door": "doh",
    "floor": "floh",
    "for": "fuh",
    "your": "yoh",
    "sure": "shoah",
    "poor": "poh",

    # R after vowel in middle
    "party": "pah-ty",
    "hard": "hahd",
    "heart": "haht",
    "star": "stah",
    "car": "cah",
    "dark": "dahk",
    "farm": "fahm",
    "arm": "ahm",
    "charm": "chahm",
    "large": "lahje",
    "garden": "gah-den",
    "market": "mah-ket",
    "start": "staht",
    "smart": "smaht",
    "sharp": "shahp",
    "mark": "mahk",
    "park": "pahk",
    "parlor": "pah-luh",

    # Broad A (trap-bath split)
    "can't": "cahnt",
    "cant": "cahnt",
    "dance": "dahnce",
    "chance": "chahnce",
    "france": "frahnce",
    "answer": "ahnsuh",
    "ask": "ahsk",
    "asked": "ahsked",
    "last": "lahst",
    "past": "pahst",
    "fast": "fahst",
    "class": "clahss",
    "glass": "glahss",
    "grass": "grahss",
    "pass": "pahss",
    "path": "pahth",
    "bath": "bahth",
    "half": "hahf",
    "laugh": "lahff",
    "staff": "stahff",
    "draft": "drahft",
    "example": "egg-zahm-pul",
    "demand": "dee-mahnd",
    "command": "kuh-mahnd",
    "plant": "plahnt",

    # Elongated/measured vowels
    "yes": "yehs",
    "no": "noh",
    "so": "soh",
    "go": "goh",
    "do": "doo",
    "who": "hoo",

    # Common transatlantic markers
    "the": "thee",
    "a": "ah",
    "of": "ahv",
    "my": "mai",
    "i": "ai",
    "my dear": "mai deah",
    "quite": "kwaite",
    "simply": "simplee",
    "frightfully": "frait-fully",
    "perfectly": "puh-fectly",
    "terribly": "tehr-ribly",
    "rather nice": "rahthuh nice",
    "darling": "dahling",
    "charming": "chahming",
    "absolutely": "absolute-ly",
    "marvelous": "mahvelous",
    "splendid": "splendidd",
    "jolly": "jolly",
    "quite so": "kwaite soh",
    "indeed": "indeedd",
    "certainly": "suh-tainly",

    # Common verbs/auxiliaries
    "are": "ah",
    "were": "wuh",
    "or": "aw",
    "for": "fuh",
    "her": "huh",
    "you're": "yoah",
    "they're": "theah",
    "we're": "weah",
    "i'm": "aim",

    # Occupation/title words
    "sir": "suh",
    "miss": "miss",
    "mister": "mistuh",
    "doctor": "dok-tuh",
    "teacher": "teachuh",

    # Time words
    "morning": "mawning",
    "evening": "eevening",
    "afternoon": "ahftuh-noon",
    "yesterday": "yestuh-day",
    "tomorrow": "tomarrow",
    "tonight": "tonaight",

    # Numbers
    "four": "foh",
    "fourth": "fohth",

    # Specific to our test phrases
    "attest": "attest",
    "hardware": "hahdwayuh",
    "epoch": "eepock",
    "laboratories": "labrah-trees",
    "machinery": "muh-sheenery",
    "antiquity": "an-tick-wity",
    "machine": "muh-sheen",
    "virtual": "vuh-tyoo-al",
    "genuine": "jen-you-ine",
    "masquerade": "mahsk-uh-rade",
    "dreadful": "dredfool",
    "attestation": "attest-ayshun",
    "evening": "eevening",
    "guide": "gaide",
    "breakthrough": "brake-throo",
    "computing": "cumpyooting",
    "foresight": "fohsight",
    "preserve": "preezuhve",
    "vintage": "vintidge",
    "bonus": "bonus",
    "rewards": "reewahds",
}


def respell_transatlantic(text):
    """Replace words with transatlantic phonetic spellings."""
    # Sort by length descending so multi-word phrases are replaced first
    keys = sorted(TRANSATLANTIC_RESPELL.keys(), key=len, reverse=True)

    result = text
    for word in keys:
        # Case-insensitive word-boundary replacement
        replacement = TRANSATLANTIC_RESPELL[word]
        pattern = r'\b' + re.escape(word) + r'\b'
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result


def main():
    import sys
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print(f"Original: {text}")
        print(f"Respelled: {respell_transatlantic(text)}")
    else:
        # Test cases
        test_cases = [
            "Good evening. I am Sophia Elya, and I shall be your guide this evening.",
            "One simply must attest ones hardware before the epoch settles, darling.",
            "And now, from the laboratories of Elyan Labs, a breakthrough in computing.",
            "The antiquity bonus rewards those with the foresight to preserve fine vintage machinery.",
            "How perfectly dreadful. A virtual machine attempting to masquerade as genuine hardware.",
            "I rather think we ought to dance at the party tomorrow evening, dear.",
            "The matter is quite simply a frightfully important consideration.",
        ]
        for t in test_cases:
            print(f"\nORIG: {t}")
            print(f"NEW:  {respell_transatlantic(t)}")


if __name__ == "__main__":
    main()

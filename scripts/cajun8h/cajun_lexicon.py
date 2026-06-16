"""
Louisiana pronunciation lexicon — respell hard place-names and Cajun lexicon
phonetically BEFORE TTS so CosyVoice2 says them the Louisiana way instead of
guessing from the (often Choctaw/Houma/French-creole) spelling.

Python is the source of truth. `respell(text)` applies it; `to_js()` emits the
same map for the website. Respellings are French-friendly first-guesses — refine
by ear (generate, listen, tweak the right-hand side). Keys match whole words,
case-insensitive; leading capitalization is preserved.
"""
import re

LEXICON = {
    # ---- place-names: parishes, towns, bayous (the tricky, non-French ones) ----
    "Opelousas":    "Opéloussa",
    "Atchafalaya":  "Atchafalaïa",
    "Natchitoches": "Nakitoche",
    "Calcasieu":    "Calcassiou",
    "Ouachita":     "Ouachitaw",
    "Tchoupitoulas":"Tchopitoulas",
    "Pontchartrain":"Pontchartrain",
    "Tangipahoa":   "Tangipahoa",
    "Plaquemines":  "Plakemine",
    "Plaquemine":   "Plakemine",
    "Thibodaux":    "Tibodo",
    "Lafourche":    "Lafourche",
    "Maurepas":     "Morepa",
    "Tchefuncte":   "Tchéfonkte",
    "Manchac":      "Manchak",
    "Catahoula":    "Catahoula",
    "Tickfaw":      "Tikfo",
    "Ponchatoula":  "Ponchatoula",
    "Houma":        "Houma",
    "Terrebonne":   "Terrebonne",
    "Vermilion":    "Vermillon",
    "Acadiana":     "Acadiana",
    "Eunice":       "Younisse",
    "Mamou":        "Mamou",
    "Carencro":     "Carencro",
    "Chackbay":     "Chakbay",
    "Cocodrie":     "Cocodri",
    "Schriever":    "Schriveur",
    "Galliano":     "Galliano",
    "Larose":       "Larose",
    "Chauvin":      "Chauvin",
    "Montegut":     "Montégu",
    "Bourg":        "Bourg",
    "Vacherie":     "Vacherie",
    "Donaldsonville":"Donaldsonville",
    "Natchez":      "Natchez",
    "Coteau":       "Coteau",
    "Teche":        "Tèche",
    "Breaux":       "Bro",
    "Boudreaux":    "Boudro",
    "Thibodeaux":   "Tibodo",
    "Hebert":       "Ébère",
    "Guidry":       "Guidri",
    # ---- Cajun / Louisiana French lexicon ----
    "maringouin":   "marin-gouin",
    "maringouins":  "marin-gouins",
    "couillon":     "couyon",
    "couillons":    "couyons",
    "lagniappe":    "lagnappe",
    "fais-do-do":   "fé-dodo",
    "bourrée":      "bourré",
    "gris-gris":    "gri-gri",
    "étouffée":     "étouffé",
    "ouaouaron":    "wawaron",
    "gratons":      "graton",
    "cracklins":    "crackline",
    "boucherie":    "boucherie",
    "traiteur":     "treteur",
    "veillée":      "veillé",
    "cocodril":     "cocodri",
    "chevrette":    "chevrette",
    "barbue":       "barbu",
    "nonc":         "nonk",
    "catin":        "catin",
    "capon":        "capon",
}

def _replace(m):
    word = m.group(0)
    repl = LEXICON.get(word) or LEXICON.get(word.lower())
    if repl is None:
        return word
    if word[:1].isupper():
        repl = repl[:1].upper() + repl[1:]
    return repl

_keys = sorted(LEXICON.keys(), key=len, reverse=True)
_pat = re.compile(r"\b(" + "|".join(re.escape(k) for k in _keys) + r")\b", re.IGNORECASE)

def respell(text: str) -> str:
    """Apply the Louisiana pronunciation lexicon to a line of text."""
    return _pat.sub(_replace, text)

def to_js() -> str:
    """Emit the lexicon + respell() as a self-contained JS snippet for the website."""
    import json
    entries = json.dumps(LEXICON, ensure_ascii=False)
    return (
        "var CAJUN_LEXICON=" + entries + ";\n"
        "function cajunRespell(t){var ks=Object.keys(CAJUN_LEXICON).sort(function(a,b){return b.length-a.length;});"
        "var re=new RegExp('\\\\b('+ks.map(function(k){return k.replace(/[.*+?^${}()|[\\]\\\\]/g,'\\\\$&');}).join('|')+')\\\\b','gi');"
        "return t.replace(re,function(w){var r=CAJUN_LEXICON[w]||CAJUN_LEXICON[w.toLowerCase()];if(!r)return w;"
        "return w[0]===w[0].toUpperCase()?r[0].toUpperCase()+r.slice(1):r;});}"
    )

if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--js"]:
        print(to_js())
    else:
        print(respell(" ".join(sys.argv[1:]) or
              "On va passer par Opelousas et l'Atchafalaya. Attention aux maringouins, cher !"))

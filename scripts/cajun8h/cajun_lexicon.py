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
    # ---- greetings / fast-speech elisions (drop the over-articulated tail) ----
    # Cajun fast speech swallows the -ment nasal: "comment ça va" said quick
    # lands on top of the surname "Comeaux" (kɔ-mo). Respell to elide so
    # CosyVoice blends it instead of enunciating the textbook ko-MAH(n).
    # fast Cajun: "comment ça" collapses to the surname Comeaux (ko-MOH) -> spell
    # it literally as the last name so CosyVoice says "ko-mo sa va", not "kom-sa-va".
    "comment ça va":  "Comeaux ça va",
    "comment ca va":  "Comeaux ça va",   # ascii-typed variant
    "comment ça":     "Comeaux",
    "comment ca":     "Comeaux",
    "quand même":     "quan mêm",     # Cajun "ka-MEM" = anyhow / anyway / all the same
    "quand meme":     "quan mêm",     # ascii-typed variant
    "comme ci comme ça": "comme ci comme ça",  # kom-see-kom-sah = so-so / middling
    "comme ci comme ca": "comme ci comme ça",  # ascii-typed variant
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
    "fais-do-do":   "fé dôh-dôh",          # fay-DOH-doh — the dance (+ "go to sleep" to the kids)
    "fais do-do":   "fé dôh-dôh",
    "fais do do":   "fé dôh-dôh",
    "fais dodo":    "fé dôh-dôh",
    "fait do do":   "fé dôh-dôh",
    "fait do-do":   "fé dôh-dôh",
    "faisdodo":     "fé dôh-dôh",
    "à soir":       "à soi",            # ah-SWAH — "tonight" (Cajun softens/drops the R)
    "a soir":       "à soi",
    "asoir":        "à soi",
    "à soir,":      "à soi,",
    "ti bébé":      "ti bébé",          # tee-bay-BAY — little baby (Cajun 'ti' = petit)
    "ti bebe":      "ti bébé",
    "t'bébé":       "ti bébé",
    "t'bebe":       "ti bébé",
    "tit bébé":     "ti bébé",
    # Cajun diminutive 'tit (from petit) = "TEET" (hard t) -> respell "tite"
    "ti fille":     "tite fille",       # teet-FEE — little girl
    "tit fille":    "tite fille",
    "'tit fille":   "tite fille",
    "teet fille":   "tite fille",
    "ti garçon":    "tite garçon",      # teet-gar-SOHN — little boy
    "tit garçon":   "tite garçon",
    "'tit garçon":  "tite garçon",
    "teet garçon":  "tite garçon",
    "tit enfant":   "tite enfant",
    "teet":         "tite",             # standalone Cajun diminutive
    # Cajun-English code-switch: t'boy / t'girl (ti-boy, ti-girl)
    "t'boy":        "tite boy",
    "ti-boy":       "tite boy",
    "teet boy":     "tite boy",
    "t'girl":       "tite girl",
    "ti-girl":      "tite girl",
    "teet girl":    "tite girl",
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
    "cocodrie":     "cocodri",         # co-co-DREE — Cajun for alligator (colonial French
    "alligator":    "cocodri",         #   saw gators, called them crocodiles — close cousin)
    "alligators":   "cocodris",
    "chevrette":    "chevrette",
    "barbue":       "barbu",
    "nonc":         "nonk",
    "catin":        "catin",
    "capon":        "capon",
    "couche-couche":"couche-couche",   # koosh-koosh — fried cornmeal breakfast
    "coush-coush":  "couche-couche",
    "cooshcoosh":   "couche-couche",
    "tracas":       "traca",           # trah-KAH — trouble / fuss / worries (silent s)
    "thraca":       "traca",
    # ---- LSU-attested everyday verbs, games & descriptors (ear-tunable) ----
    # Source: LSU Cajun French-English glossary, bracketed pronunciation cues.
    "babiller":     "bah-bi-yé",       # [BAH-BEE-YEH] — to scold, usually a child
    "badgeuler":    "bah-jeu-lé",      # [BAH JEUH LEH] — to harass / nag
    "badjeuler":    "bah-jeu-lé",      # LSU spelling variant for badgeuler
    "cacher-faite": "kah-shé-fête",    # [KAH SHEH FET] — hide-and-seek game
    "cachez-fête":  "kah-shé-fête",    # LSU variant spelling
    "cache-et-fête":"kah-shé-fête",    # LSU variant spelling
    "brailler":     "brah-yé",         # [BRAH YEH] — to cry
    "brin":         "bren",            # [BREn] — a little bit / small quantity
    "fouiller":     "fou-yé",          # [FOOYEH] — to dig / rummage around
    "fréquenter":   "fré-kan-té",      # [FREH KAnTEH] — to socialize / visit regularly
    "frequenter":   "fré-kan-té",      # ascii-typed variant
    "gaspiller":    "gah-spi-yé",      # [GAH SPEE YEH] — to waste
    "grouiller":    "grou-yé",        # [GROOYEH] — to move
    "fatras":       "fah-trah",        # [FAH TRAH] — trash / no-account person
    "pistache":     "pi-stahsh",       # [PEE STAH SH] — peanut
    "plaquemine":   "plahk-mine",      # [PLAHKMEEN] — persimmon; lowercase lexicon sense
    "poulailler":   "pou-la-yé",       # [POO LAH YEH] — chicken coop
    "quereller":    "kré-lé",          # [KRE LEH] — to argue / scold
    "espeler":      "esplé",           # [ESPLEH] — to spell
    "dételer":      "dé-tlé",          # [DEH TLEH] — to unharness
    "deteler":      "dé-tlé",          # ascii-typed variant
    "avalasse":     "ah-vah-lahs",     # [AH VAH LAHS] — downpour
    # ---- St. Landry / prairie-Cajun colloquial (deep-research 2026-06-16) ----
    "asteur":       "asteur",          # ah-STUR — "now" (from à cette heure)
    "astheure":     "asteur",
    "à cette heure":"asteur",
    "quoi faire":   "quoi faire",      # kwa-FAIR / ko-FAIR — "why"
    "kofaire":      "quoi faire",
    "quo'faire":    "quoi faire",
    "mais là":      "mè là",           # may-LAH — "well now / well then"
    "chaoui":       "chawi",           # sha-WEE — raccoon (Choctaw)
    "chaouis":      "chawis",
    "tayau":        "tayo",            # ta-YO — hound dog
    "tayaus":       "tayos",
    "gru":          "grou",            # groo — grits
    "cabri":        "cabri",           # ka-BREE — goat
    "chaudin":      "chaudin",         # sho-DAN — stuffed pig stomach
    "quelque chose":"quéque chose",    # kek-SHOWZ — Cajun contraction of "something"
    "quelque":      "quéque",
    "quelqu'un":    "quéqu'un",
    "déboires":     "débouare",        # day-BWAR — woes/setbacks; "tracas et déboires" = real trouble
    "deboires":     "débouare",
    "deba":         "déba",            # day-BAH — maw-maw's clipped "déboires"
    "gremise":      "grémise",         # greh-MEEZ — grime / filth / "dirty dirt" (ear-tunable)
    "grémise":      "grémise",
    "gradou":       "gradou",          # grah-DOO — crud / grime / gunk (pairs w/ gremise)
    "gradoux":      "gradou",
    "gradu":        "gradou",
    # ---- folklore, places & culture ----
    "rougarou":     "rougarou",        # roo-gah-ROO — Cajun werewolf (loup-garou)
    "rogaroux":     "rougarou",
    "rougaroux":    "rougarou",
    "loup-garou":   "loup-garou",
    "bétaille":     "bataï",           # bah-TIE — critter / varmint / monster
    "bétailles":    "bataïs",
    "betaille":     "bataï",
    "betailles":    "bataïs",
    "batai":        "bataï",
    "batais":       "bataïs",
    "Grand Caillou":"Grand Caïou",     # grahn-KYE-oo — Terrebonne community
    "Caillou":      "Caïou",
    "Magnalite":    "Magnalite",       # cast-aluminum pots in every Cajun kitchen
    "gardez-moi ça":"gadez don",       # fast Cajun: "look at that!" -> ga-DAY don
    "garde-moi ça": "gadez don",
    "garde moi ça": "gadez don",
    "cher":         "cha",             # term of endearment — "cha" (a as in 'at'), not "shah"
    "chère":        "cha",
    "chere":        "cha",             # ascii-typed variant
    "Cher":         "Cha",
    # ---- old Cajun given names (ear-tunable) ----
    "Aleda":        "Aléda",
    "Adalaya":      "Adalaïa",
    "Sédonie":      "Sédoni",
    "Sedonie":      "Sédoni",
    "Eulalie":      "Eulalie",
    "Octave":       "Octave",
    "Adelard":      "Adélar",
    "Ozémé":        "Ozémé",
    "Augustin":     "Augustin",
    "Remi":         "Rémi",
    "Attakapas":    "Atakapa",
    "Acadie":       "Acadie",
    # ---- Sophia's own name, the Cajun way (ear-tunable) ----
    "Sophia Elya":  "Sofia Élia",
    "Elya":         "Élia",
    "Sophia":       "Sofia",
}

_lower_index = {k.lower(): v for k, v in LEXICON.items()}

def _replace(m):
    word = m.group(0)
    repl = LEXICON.get(word) or _lower_index.get(word.lower())
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
        "var CAJUN_LEX_LC={};Object.keys(CAJUN_LEXICON).forEach(function(k){CAJUN_LEX_LC[k.toLowerCase()]=CAJUN_LEXICON[k];});\n"
        "function cajunRespell(t){var ks=Object.keys(CAJUN_LEXICON).sort(function(a,b){return b.length-a.length;});"
        "var re=new RegExp('\\\\b('+ks.map(function(k){return k.replace(/[.*+?^${}()|[\\]\\\\]/g,'\\\\$&');}).join('|')+')\\\\b','gi');"
        "return t.replace(re,function(w){var r=CAJUN_LEXICON[w]||CAJUN_LEX_LC[w.toLowerCase()];if(!r)return w;"
        "return w[0]===w[0].toUpperCase()?r[0].toUpperCase()+r.slice(1):r;});}"
    )

if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--js"]:
        print(to_js())
    else:
        print(respell(" ".join(sys.argv[1:]) or
              "On va passer par Opelousas et l'Atchafalaya. Attention aux maringouins, cher !"))

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
    # ---- Acadiana town-name respellings (municipal/place names; ear-tunable) ----
    "Ville Platte": "Ville Plat",      # Evangeline Parish city; final -e is not voiced
    "Gueydan":      "Guédan",          # Vermilion Parish town; helps avoid hard English "guy-dan"
    "Iota":         "Aïota",           # Acadia Parish town; keep the opening eye sound
    "Duson":        "Duzon",           # Lafayette/Acadia Parish town
    "Erath":        "Érat",            # Vermilion Parish town
    "Delcambre":    "Delcambe",        # Iberia/Vermilion Parish town
    "Mermentau":    "Mermento",        # Acadia Parish village / river name
    "Meraux":       "Méro",            # St. Bernard Parish community; French -aux ending
    "Pierre Part":  "Pierre Par",      # Assumption Parish community; final -t is silent
    "Basile":       "Bazile",          # Evangeline/Acadia Parish town
    "Rayne":        "Rain",            # Acadia Parish city
    "Kaplan":       "Caplan",          # Vermilion Parish city
    "Abbeville":    "Abb-ville",       # Vermilion Parish city; cue the local clipped middle
    "Church Point": "Church Pointe",   # Acadia Parish town; keep the place-name cadence
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
    # ---- bayou boats, water & fishing-work terms (LSU glossary + public LA travel glossaries) ----
    "bayou":        "baïou",           # Visit Baton Rouge: bayou = slow-moving stream, "bye-you"
    "bayous":       "baïous",
    "pirogue":      "pi-ro",           # Explore Louisiana/Acadiana Table: Cajun canoe, PEE-row
    "pirogues":     "pi-ros",
    "à la pêche":   "à la pèch",       # LSU Cajun glossary: pêche = fishing; final e softened
    "a la peche":   "à la pèch",
    "pêche":        "pèch",
    "peche":        "pèch",
    "eau douce":    "eau dous",        # LSU glossary: eau douce = fresh water, douce [DOOS]
    "eaux douces":  "eaux dous",
    "eau salée":    "eau salé",        # LSU glossary: eau salée = salt water; final e not over-read
    "eau salee":    "eau salé",
    "eaux salées":  "eaux salés",
    "eaux salees":  "eaux salés",
    "ecrevisse":    "écrevisse",       # LSU glossary: écrevisse = crawfish; keep accent in generated text
    "ecrevisses":   "écrevisses",
    "crevette":     "chevrette",       # LSU glossary notes Louisiana French chevrette for shrimp
    "crevettes":    "chevrettes",
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
    # ---- Louisiana/Cajun culinary pronunciations (public glossary attested) ----
    "boudin":       "bou-dan",         # Source: Visit Baton Rouge, Boudin (boo-dan)
    "boudins":      "bou-dans",        # plural; same attestation as boudin
    "beignet":      "bèn-yé",          # Source: Visit Baton Rouge, Beignet (bin-yay)
    "beignets":     "bèn-yés",
    "courtbouillon":"cou-bouyon",      # Source: Explore Louisiana, Courtbouillon [coo-boo-yon]
    "court bouillon":"cou-bouyon",
    "bouillie":     "bou-yee",         # Source: Acadiana Table, Bouillie (BOO yee)
    "tarte à la bouillie": "tarte à la bou-yee",
    "tarte a la bouillie": "tarte à la bou-yee",
    "boulette":     "boulet",          # Source: Acadiana Table, Boulette (BOO let)
    "boulettes":    "boulets",
    "roux":         "rou",             # Source: Big Easy Foods, Roux (roo)
    "grillades":    "gri-yades",       # Source: Big Easy Foods, Grillades (GREE yads)
    "amandine":     "arman-dine",      # Source: Big Easy Foods, Amandine (ar-man-deen)
    "au gratin":    "ô gratin",        # Source: Acadiana Table, Au gratin (oh GRAH tan)
    "cochon de lait":"cou-shon de lait", # Source: Big Easy Foods, Cochon de Lait (coo-shon duh lay)
    "café au lait": "café ô lait",     # Source: Acadiana Table, Café au Lait (caf AY oh LAY)
    "cafe au lait": "café ô lait",
    "rémoulade":    "rémoularde",      # Source: Big Easy Foods, Remoulade (rem-oo-lard)
    "remoulade":    "rémoularde",
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
    # ---- Cajun kitchen / table (Daigle, A Dict. of the Cajun Language 1984;
    #      Valdman et al., Dict. of Louisiana French 2010 — ear-tunable) ----
    "jambalaya":     "jambalaïa",      # jahm-bah-LIE-ah (same -aya->-aïa as Atchafalaya)
    "andouille":     "andouï",         # ahn-DOO-ee — smoked pork sausage (drop final schwa)
    "andouilles":    "andouïs",
    "maque choux":   "mak-chou",       # mock-SHOO — stewed corn dish
    "sauce piquante":"sauce picante",  # sohss pee-KAHNT — tomato-pepper stew
    "fricassée":     "fricassé",       # free-kah-SAY (same -ée->-é as étouffée->étouffé)
    "filé":          "filé",           # FEE-lay — ground sassafras for gumbo
    "tasso":         "tasso",          # TAH-so — cured smoked pork
    # ---- everyday Cajun French (Daigle 1984; Valdman DLF 2010 — ear-tunable) ----
    "mais oui":      "mè oui",         # may-WEE — "why yes / of course" (same mè- as "mais là")
    "mais non":      "mè non",         # may-NOHN — "why no"
    "icitte":        "icitte",         # ee-SEET — Cajun "here" (vs. textbook "ici")
    "couillonnade":  "couyonade",      # koo-yoh-NAHD — foolishness (same root as couillon->couyon)
    "couillonnades": "couyonades",
    "bébête":        "bébête",         # bay-BET — critter / bug / bogeyman
    "bébêtes":       "bébêtes",
    "tonnerre":      "tonnerre",       # toh-NAIR — "thunder", common exclamation ("tonnerre m'écrase")
    # ---- more place-names (USGS GNIS; local Louisiana French pronunciation) ----
    "Des Allemands": "Dèz Almand",     # day-zal-MAHN — St. Charles/Lafourche town ("the Germans")
    "Choupique":     "Choupique",      # shoo-PEEK — bayou/community + the bowfin fish
    "Grosse Tête":   "Grosse Tête",    # grohs-TET — Iberville Parish town
    "Arnaudville":   "Arnaudville",    # ar-no-VEEL — St. Landry/St. Martin town
    "Bogalusa":      "Bogalouza",      # boh-gah-LOO-zah — Washington Parish town
    # ---- Sophia's own name, the Cajun way (ear-tunable) ----
    "Sophia Elya":  "Sofia Élia",
    "Elya":         "Élia",
    "Sophia":       "Sofia",
    # ---- Cajun music, dance & Mardi Gras (attest.: Valdman et al.,
    #      "Dictionary of Louisiana French as Spoken in Cajun, Creole and
    #      American Indian Communities", 2010; Ancelet, "Cajun & Creole Music
    #      Makers") — respell so CosyVoice drops the textbook tails ----
    "frottoir":     "frotwar",         # fro-TWAR — zydeco rubboard (oi -> wa, silent r)
    "frottoirs":    "frotwars",
    "'tit fer":     "tit fèr",         # tee-FAIR — the triangle, the "little iron"
    "tit fer":      "tit fèr",
    "Courir de Mardi Gras": "Couri d'Mardi Gras",  # koo-REE — the rural Mardi Gras run
    "courir":       "couri",           # koo-REE — to run (Cajun drops the final r)
    "capuchon":     "capichon",        # ka-pee-SHAWN — conical Mardi Gras hat
    "capuchons":    "capichons",
    "la-la":        "la-la",           # lah-lah — old Creole house dance (zydeco's root)
    "valse":        "vals",            # vahls — waltz (silent final e)
    "valses":       "vals",
    # ---- swamp critters & nature (attest.: Dictionary of Louisiana French, 2010) ----
    "chevreuil":    "chevreuye",       # shev-RUH-y — deer
    "chevreuils":   "chevreuyes",
    "écureuil":     "écureuye",        # ay-ku-RUH-y — squirrel
    "écureuils":    "écureuyes",
    "bec-croche":   "bèk-croche",      # bek-KROSH — ibis ("crooked beak")
    "bec-croches":  "bèk-croches",
    "bête puante":  "bèt pwante",      # bet-PWANT — skunk ("stinking beast")
    "mulet":        "mulè",            # mu-LAY — mullet (silent t)
    "soulard":      "soula",           # soo-LAH — drunkard (silent r-d)
    # ---- kinship & terms of address (Daigle, "A Dictionary of the Cajun
    #      Language" 1984; Valdman et al., "Dictionary of Louisiana French" 2010
    #      — ear-tunable) ----
    "parrain":      "parran",      # pa-RAN — godfather (drop the textbook -ain glide)
    "marraine":     "marenne",     # ma-REN — godmother
    "nénaine":      "nénenne",     # nay-NEN — godmother (Cajun affectionate form)
    "mononc":       "mononk",      # moh-NONK — uncle ("mon oncle" fused; cf. "nonc")
    "matante":      "matante",     # ma-TANT — auntie ("ma tante" fused)
    "popa":         "popa",        # poh-PAH — papa / dad
    "moman":        "moman",       # moh-MAHN — mama / mom
    "bougre":       "bougue",      # BOOG — fellow / guy (drop the -re)
    "bougres":      "bougues",
    "vaillant":     "vayan",       # vah-YAHN — brave, hardworking (a term of praise)
    "vaillante":    "vayante",
    # ---- prairie, marsh & bayou land (USGS GNIS feature terms; Valdman DLF 2010) ----
    "coulée":       "coulé",       # koo-LAY — gully / seasonal stream (cf. étouffée->étouffé)
    "coulées":      "coulés",
    "platin":       "platin",      # plah-TANH — low wet spot that holds water in a field
    "trainasse":    "trénasse",    # tray-NAHSS — trapper's narrow channel cut through marsh
    "trénasse":     "trénasse",    # spelling variant
    "batture":      "batture",     # bah-TUR — land between the levee and the river
    "chénière":     "chénière",    # shay-NYAIR — live-oak ridge in the marsh (Chénière au Tigre)
    "cheniere":     "chénière",    # ascii-typed variant
    "marais":       "marè",        # mah-REH — marsh (silent -s)
    "anse":         "anse",        # AHNSS — cove / inlet (Anse La Butte, Grand Anse)
    # ---- fish & trapping animals (Daigle 1984; Valdman DLF 2010 — ear-tunable) ----
    "sac-à-lait":   "sac-à-lè",    # sak-ah-LEH — white crappie ("milk sack")
    "sac à lait":   "sac-à-lè",    # spacing variant
    "patassa":      "patassa",     # pah-tah-SAH — sunfish / perch
    "rat de bois":  "rat d'bois",  # rah-d-BWAH — opossum ("woods rat")
    "rat musqué":   "rat musqué",  # rah moos-KAY — muskrat (the prized trapping animal)
    "tortue":       "tortu",       # tor-TU — turtle (silent -e)
    "tortues":      "tortus",
    # ---- religion, holidays & the faith ----
    "Toussaint":       "Toussan",      # All Saints' Day, "too-SAN" — Valdman, Dict. of La. French (DLF) 2010
    "Pâques":          "Pâk",          # Easter, "pahk" (final -s silent) — DLF
    "Paques":          "Pâk",          # ascii-typed variant
    "Carême":          "Carèm",        # Lent, "ka-REM" — DLF
    "Careme":          "Carèm",        # ascii-typed variant
    "réveillon":       "révéyon",      # Christmas / New-Year night feast, "ray-vay-YOHN" — DLF
    "reveillon":       "révéyon",      # ascii-typed variant
    "Vendredi Saint":  "Vendredi San", # Good Friday, "vahn-druh-dee-SAN" — DLF
    "messe de minuit": "mess de minwi",# midnight Mass, "mess-duh-mee-NWEE" — Ancelet
    "bon Dieu":        "bon Djeu",     # "good Lord" — Cajun palatalizes Dieu to "djeu" — Ancelet
    "Mardi Gras":      "Mardi Gra",    # "mar-dee-GRAH" (final -s silent) — DLF
    # ---- weather, water & the wild ----
    "suroît":          "surwa",        # southwest wind, "su-RWA" (oî->wa, final t silent) — DLF
    "suroit":          "surwa",        # ascii-typed variant
    "crapaud":         "crapo",        # toad/frog, "kra-PO" — Daigle 1984
    "crapauds":        "crapo",
    "mouche à feu":    "mouch à feu",  # firefly, "moosh-a-FEU" — DLF
    "au ras":          "o ra",         # right beside / close to, "o-RA" — DLF
    # ---- time & everyday turns of phrase ----
    "tantôt":          "tantô",        # a while ago / later on, "tahn-TOH" — DLF
    "tantot":          "tantô",        # ascii-typed variant
    "bêtise":          "bétiz",        # nonsense / foolishness, "bay-TEEZ" — DLF
    "betise":          "bétiz",        # ascii-typed variant
    "lâche pas la patate": "lâche pa la patat",  # "don't drop the potato" = don't give up — Ancelet
    "lache pas la patate": "lâche pa la patat",  # ascii-typed variant
    # ---- Cajun / Acadian surnames (Louisiana surname rankings + Acadian family lists) ----
    "Comeaux":      "Como",            # -eaux family-name ending: enforce the Cajun "o" sound
    "Arceneaux":    "Arsenô",          # common Acadian/Cajun surname; same -eaux -> o pattern
    "Robichaux":    "Robicho",
    "Broussard":    "Broussar",        # final consonant softened/silent in French-friendly TTS
    "Fontenot":     "Fontenô",         # Louisiana surname; final -ot rendered as open "o"
    "Landry":       "Landri",
    "LeBlanc":      "Le Blan",         # final c silent in French-style surname pronunciation
    "Dugas":        "Duga",            # final s silent
    "Doucet":       "Doucé",           # -cet family-name ending, closer to "doo-say"
    "Bourgeois":    "Bourjwa",         # French oi -> wa
    "Richard":      "Richar",          # final d silent
    "Thériot":      "Tério",
    "Theriot":      "Tério",
    "Sonnier":      "Sonyé",
    "Melançon":     "Mélançon",
    "Melancon":     "Mélançon",
    # ---- LSU glossary household, clothing & everyday descriptor terms ----
    # Source: LSU Cajun French-English glossary, entries with bracketed pronunciation cues.
    "camisole":     "camizole",        # [KAH MEE ZAWL] — nightgown; z-like s keeps the local cue
    "canaille":     "canaï",           # [KAH NAHY] — mischievous / wily, softened local sense
    "canaillerie":  "canaïrie",        # [KAH NAH YREE] — mischief / trickery
    "canique":      "canique",         # [KAH NEEK] — marble (child's toy)
    "capot":        "capo",            # [KAPO] — coat / jacket, final t silent
    "capot ciré":   "capo ciré",       # raincoat; same final-t drop
    "carrément":    "carrèman",        # [KAH REH MAn] — directly / abruptly
    "carrement":    "carrèman",        # ascii-typed variant
    "chapelet":     "chaplé",          # [SHAH PLEH] — rosary
    "chaste-femme": "chasse-femme",    # [SHAHS FAM] — midwife; LSU gives this pronunciation variant
    "hameçon":      "am-son",          # [AHM SOn] — fish hook, initial h silent
    "hamecon":      "am-son",          # ascii-typed variant
    "hélas":        "éla",             # [EHLAH] — sigh
    "helas":        "éla",             # ascii-typed variant
    "grands hélas": "grands éla",      # LSU phrase: faire des grands hélas
    "grands helas": "grands éla",
    "hibou":        "ibou",            # [EEBOO] — owl, initial h silent
    "hormis que":   "ormi que",        # [AWRMEE KEUH] — unless
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

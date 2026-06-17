"""Held-out Cajun French eval set — representative, covers the hard words
(endearment, food, folklore, place-names, colloquial) and a range of length.
These are NOT verbatim training lines. `text` is the intended Cajun French;
the harness feeds respell(text) to the model and ASR-compares against the fed text."""

SENTENCES = [
    # greetings / endearment
    ("greet1", "Mais comment ça va, cher? Ça fait longtemps que je t'ai pas vu."),
    ("greet2", "Bonjour mon ami, viens t'asseoir, on va jaser un peu."),
    ("endear1", "Dors bien, ti bébé, maman est icitte avec toi."),
    # food
    ("food1", "On va faire un bon gombo avec du riz et de la chevrette ce soir."),
    ("food2", "Maw-maw fait la couche-couche le matin dans la grande Magnalite."),
    ("food3", "Passe-moi l'étouffée, cher, et un peu de pain français."),
    # folklore
    ("folk1", "Le rougarou est sorti à soir près de Grand Caillou, fais attention."),
    ("folk2", "Les bétailles et les cocodries restent dans le marais, pas dans la maison."),
    ("folk3", "Quand t'es pas sage, le rougarou va venir te chercher, gardez-moi ça."),
    # place-names (the tricky non-French ones)
    ("place1", "On a passé par Opelousas et l'Atchafalaya pour aller à Lafayette."),
    ("place2", "Le bayou Teche coule à travers la paroisse Saint-Landry."),
    ("place3", "De Calcasieu jusqu'à Natchitoches, c'est tout la Louisiane française."),
    # colloquial
    ("coll1", "Asteur, quoi faire tu veux pas manger, toi? T'as pas faim?"),
    ("coll2", "Mais là, c'est rien que des tracas et des déboires aujourd'hui."),
    ("coll3", "Lâche pas la patate, cher, on va trouver une manière."),
    # longer / prosody
    ("long1", "Mon grand-père parlait rien que le français, et il racontait des contes "
              "le soir sur la galerie pendant que les maringouins chantaient."),
    ("long2", "Le français de la Louisiane, c'est notre héritage; faut pas le laisser "
              "mourir avec la vieille génération, cher."),
    ("long3", "On va tous au fais do-do à soir, il va avoir de la musique, "
              "du manger, et tout le monde va danser jusqu'au matin."),
]

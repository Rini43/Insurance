import spacy


MODEL_PATH = "models/vehicle_damage_ner"


nlp = spacy.load(MODEL_PATH)


text = """
The vehicle collided with another car.
The front bumper was severely damaged.
The left headlight was cracked and the bonnet had a large dent.
"""


doc = nlp(text)


print("\nExtracted entities:\n")

for ent in doc.ents:

    print(
        f"{ent.text:30} -> {ent.label_}"
    )

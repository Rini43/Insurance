import json
import random
import spacy

from spacy.training import Example
from spacy.util import minibatch, compounding


DATA_FILE = "data/training_data.json"
MODEL_DIR = "models/vehicle_damage_ner"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


# --------------------------------------------------
# Create blank English NLP pipeline
# --------------------------------------------------

nlp = spacy.blank("en")

ner = nlp.add_pipe("ner")


# --------------------------------------------------
# Add entity labels
# --------------------------------------------------

labels = [
    "DAMAGE_PART",
    "DAMAGE_TYPE",
    "DAMAGE_SEVERITY",
    "ACCIDENT_EVENT",
    "CAUSE"
]

for label in labels:
    ner.add_label(label)


# --------------------------------------------------
# Convert training data to spaCy Examples
# --------------------------------------------------

examples = []

for item in data:

    text = item["text"]
    entities = item["entities"]

    doc = nlp.make_doc(text)

    example = Example.from_dict(
        doc,
        {
            "entities": entities
        }
    )

    examples.append(example)


# --------------------------------------------------
# Train
# --------------------------------------------------

optimizer = nlp.begin_training()

for epoch in range(30):

    random.shuffle(examples)

    losses = {}

    batches = minibatch(
        examples,
        size=compounding(
            4.0,
            32.0,
            1.5
        )
    )

    for batch in batches:

        nlp.update(
            batch,
            drop=0.25,
            losses=losses
        )

    print(
        f"Epoch {epoch + 1}:",
        losses
    )


# --------------------------------------------------
# Save model
# --------------------------------------------------

nlp.to_disk(MODEL_DIR)

print(f"Model saved to {MODEL_DIR}")

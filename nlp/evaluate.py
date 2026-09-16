import json
import spacy

from spacy.training import Example
from spacy.scorer import Scorer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "models/vehicle_damage_ner"
TEST_FILE = "data/processed/test.json"


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("Loading model...")

nlp = spacy.load(MODEL_PATH)


# --------------------------------------------------
# Load test data
# --------------------------------------------------

with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)


print(f"Test examples: {len(test_data)}")


# --------------------------------------------------
# Create evaluation examples
# --------------------------------------------------

examples = []

for item in test_data:

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
# Evaluate
# --------------------------------------------------

scorer = Scorer()

scores = scorer.score(examples)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\n==============================")
print("NER EVALUATION")
print("==============================")

print(
    f"Precision : {scores['ents_p']:.4f}"
)

print(
    f"Recall    : {scores['ents_r']:.4f}"
)

print(
    f"F1 Score  : {scores['ents_f']:.4f}"
)


# --------------------------------------------------
# Per-entity results
# --------------------------------------------------

print("\nPer Entity Results:")
print("------------------------------")

for label, values in scores["ents_per_type"].items():

    precision = values["p"]
    recall = values["r"]
    f1 = values["f"]

    print(
        f"{label:20}"
        f"P={precision:.4f} "
        f"R={recall:.4f} "
        f"F1={f1:.4f}"
    )

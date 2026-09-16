import json
import random
from pathlib import Path

from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_FILE = "data/annotations.json"

OUTPUT_DIR = Path("data/processed")

TRAIN_FILE = OUTPUT_DIR / "train.json"
DEV_FILE = OUTPUT_DIR / "dev.json"
TEST_FILE = OUTPUT_DIR / "test.json"

RANDOM_STATE = 42


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load annotations
# --------------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


print(f"Total examples: {len(data)}")


# --------------------------------------------------
# Remove invalid examples
# --------------------------------------------------

clean_data = []

for item in data:

    text = item.get("text", "")
    entities = item.get("entities", [])

    if not text.strip():
        continue

    valid_entities = []

    for entity in entities:

        start = entity[0]
        end = entity[1]
        label = entity[2]

        # Check boundaries
        if start < 0:
            continue

        if end > len(text):
            continue

        if start >= end:
            continue

        valid_entities.append(
            [start, end, label]
        )

    clean_data.append(
        {
            "text": text,
            "entities": valid_entities
        }
    )


data = clean_data


print(f"Valid examples: {len(data)}")


# --------------------------------------------------
# Shuffle
# --------------------------------------------------

random.seed(RANDOM_STATE)

random.shuffle(data)


# --------------------------------------------------
# Train / temporary split
# 70% train
# 30% temporary
# --------------------------------------------------

train_data, temp_data = train_test_split(
    data,
    test_size=0.30,
    random_state=RANDOM_STATE
)


# --------------------------------------------------
# Validation / test split
# 15% dev
# 15% test
# --------------------------------------------------

dev_data, test_data = train_test_split(
    temp_data,
    test_size=0.50,
    random_state=RANDOM_STATE
)


# --------------------------------------------------
# Save
# --------------------------------------------------

with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)


with open(DEV_FILE, "w", encoding="utf-8") as f:
    json.dump(dev_data, f, indent=2, ensure_ascii=False)


with open(TEST_FILE, "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)


# --------------------------------------------------
# Print statistics
# --------------------------------------------------

print("\nDataset split:")
print(f"Train: {len(train_data)}")
print(f"Dev:   {len(dev_data)}")
print(f"Test:  {len(test_data)}")

print("\nFiles created:")
print(TRAIN_FILE)
print(DEV_FILE)
print(TEST_FILE)

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_FILE = BASE_DIR / "processed" / "chunks" / "all_chunks.json"
VECTOR_DB_DIR = BASE_DIR / "processed" / "vector_db"

COLLECTION_NAME = "insurance_policies"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found:\n{CHUNKS_FILE}"
        )

    with CHUNKS_FILE.open(
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(f)

    if not chunks:
        raise ValueError("No chunks found in all_chunks.json")

    return chunks


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model():

    print()
    print("=" * 80)
    print("LOADING EMBEDDING MODEL")
    print("=" * 80)
    print(f"Model: {EMBEDDING_MODEL}")
    print()

    model = SentenceTransformer(EMBEDDING_MODEL)

    print("✓ Embedding model loaded.")

    return model


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

def create_vector_database(chunks, model):

    VECTOR_DB_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 80)
    print("CREATING VECTOR DATABASE")
    print("=" * 80)
    print(f"Database directory: {VECTOR_DB_DIR}")
    print(f"Collection: {COLLECTION_NAME}")
    print()

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR)
    )

    # Delete existing collection if present.
    # This makes the script safe to rerun after changing chunks.
    try:
        client.delete_collection(
            name=COLLECTION_NAME
        )
        print("✓ Existing collection removed.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Insurance policy and claim procedure knowledge base",
            "embedding_model": EMBEDDING_MODEL
        }
    )

    print("✓ ChromaDB collection created.")
    print()

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:

        chunk_id = chunk["chunk_id"]
        content = chunk["content"]

        metadata = {
            "document": str(chunk.get("document", "")),
            "company": str(chunk.get("company", "")),
            "page_start": int(chunk.get("page_start", 0)),
            "page_end": int(chunk.get("page_end", 0)),
            "chunk_number": int(chunk.get("chunk_number", 0)),
            "word_count": int(chunk.get("word_count", 0))
        }

        ids.append(chunk_id)
        documents.append(content)
        metadatas.append(metadata)

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    print("Generating embeddings...")
    print(f"Chunks to embed: {len(documents)}")
    print()

    embeddings = model.encode(
        documents,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print()
    print("✓ Embeddings generated.")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # --------------------------------------------------------
    # Store in ChromaDB
    # --------------------------------------------------------

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )

    print("✓ Chunks stored in ChromaDB.")

    return collection


# ============================================================
# VALIDATE DATABASE
# ============================================================

def validate_database(collection, chunks):

    print()
    print("=" * 80)
    print("VECTOR DATABASE VALIDATION")
    print("=" * 80)

    count = collection.count()

    print(f"Expected chunks : {len(chunks)}")
    print(f"Stored vectors  : {count}")

    if count != len(chunks):
        raise RuntimeError(
            f"Vector count mismatch. "
            f"Expected {len(chunks)}, got {count}."
        )

    # Test retrieving one stored record
    result = collection.get(
        limit=1
    )

    if not result["ids"]:
        raise RuntimeError(
            "Database validation failed: no records returned."
        )

    print("✓ Vector count validation passed.")
    print("✓ Database retrieval test passed.")
    print()
    print("RESULT: PASS")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("INSURANCE KNOWLEDGE BASE - VECTOR DATABASE BUILDER")
    print("=" * 80)

    print()
    print(f"Chunks file: {CHUNKS_FILE}")

    # Load chunks
    chunks = load_chunks()

    print()
    print(f"✓ Loaded {len(chunks)} chunks.")

    # Load embedding model
    model = load_embedding_model()

    # Create vector DB
    collection = create_vector_database(
        chunks,
        model
    )

    # Validate
    validate_database(
        collection,
        chunks
    )

    print()
    print("=" * 80)
    print("VECTOR DATABASE BUILD COMPLETE")
    print("=" * 80)
    print(f"Documents/chunks processed : {len(chunks)}")
    print(f"Vector database             : {VECTOR_DB_DIR}")
    print(f"Collection                  : {COLLECTION_NAME}")
    print(f"Embedding model             : {EMBEDDING_MODEL}")
    print()
    print("✓ Knowledge base is ready for semantic search.")


if __name__ == "__main__":
    main()

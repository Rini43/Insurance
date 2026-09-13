import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_DB_DIR = BASE_DIR / "processed" / "vector_db"
CHUNKS_FILE = BASE_DIR / "processed" / "chunks" / "all_chunks.json"

COLLECTION_NAME = "insurance_policies"
MODEL_NAME = "all-MiniLM-L6-v2"


print("=" * 80)
print("INSURANCE KNOWLEDGE BASE - SEMANTIC SEARCH")
print("=" * 80)

print("\nLoading embedding model...")
model = SentenceTransformer(MODEL_NAME)
print("✓ Embedding model loaded.")

print("\nConnecting to ChromaDB...")

client = chromadb.PersistentClient(
    path=str(VECTOR_DB_DIR)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print(f"✓ Collection: {COLLECTION_NAME}")
print(f"✓ Vectors: {collection.count()}")

with CHUNKS_FILE.open("r", encoding="utf-8") as f:
    chunks = json.load(f)

chunk_lookup = {
    chunk["chunk_id"]: chunk
    for chunk in chunks
}


def search_knowledge_base(query, top_k=5, company=None):

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": [
            "documents",
            "metadatas",
            "distances"
        ]
    }

    if company:
        kwargs["where"] = {
            "company": company
        }

    results = collection.query(**kwargs)

    retrieved = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        retrieved.append({
            "chunk_id": metadata.get("chunk_id"),
            "document": metadata.get("document"),
            "company": metadata.get("company"),
            "page_start": metadata.get("page_start"),
            "page_end": metadata.get("page_end"),
            "chunk_number": metadata.get("chunk_number"),
            "distance": distance,
            "content": document
        })

    return retrieved


def print_results(query, results):

    print("\n" + "=" * 80)
    print("QUERY")
    print("=" * 80)
    print(query)

    print("\n" + "=" * 80)
    print(f"TOP {len(results)} RESULTS")
    print("=" * 80)

    if not results:
        print("No results found.")
        return

    for i, result in enumerate(results, 1):

        print("\n" + "-" * 80)
        print(f"RESULT {i}")
        print("-" * 80)

        print(f"Company  : {result['company']}")
        print(f"Document : {result['document']}")
        print(
            f"Pages    : "
            f"{result['page_start']} - {result['page_end']}"
        )
        print(f"Chunk    : {result['chunk_number']}")
        print(f"Distance : {result['distance']:.4f}")

        print("\nContent:")
        print(result["content"])


def main():

    print("\nSemantic search ready.")
    print("Type 'exit' to quit.")

    while True:

        query = input("\nAsk a question: ").strip()

        if query.lower() in ["exit", "quit", "q"]:
            break

        if not query:
            continue

        results = search_knowledge_base(
            query,
            top_k=5
        )

        print_results(
            query,
            results
        )


if __name__ == "__main__":
    main()

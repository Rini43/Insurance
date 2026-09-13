import requests

from search_knowledge_base import search_knowledge_base


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

OLLAMA_TIMEOUT = 180

TOP_K = 2

MAX_CHARS_PER_CHUNK = 2500


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for i, result in enumerate(results, 1):

        content = result["content"]

        # Prevent sending unnecessarily huge chunks
        if len(content) > MAX_CHARS_PER_CHUNK:
            content = content[:MAX_CHARS_PER_CHUNK] + "\n[Content truncated]"

        source = (
            f"Source {i}\n"
            f"Company: {result['company']}\n"
            f"Document: {result['document']}\n"
            f"Pages: {result['page_start']}-{result['page_end']}\n"
            f"Content:\n{content}"
        )

        context_parts.append(source)

    return "\n\n".join(context_parts)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, results):

    context = build_context(results)

    prompt = f"""
You are an insurance policy knowledge assistant.

Answer the user's question ONLY using the policy information
provided in the context.

Rules:
1. Do not invent information.
2. Do not assume coverage that is not stated.
3. If the answer is not present, say:
   "I could not find this information in the provided policy documents."
4. Clearly distinguish coverage, exclusions, conditions and procedures.
5. Mention the insurance company and policy document when useful.
6. Include page references from the supplied sources.
7. Keep the answer concise.
8. Use simple language.

POLICY CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.1,
                "num_ctx": 2048,
                "num_predict": 250
            }
        },
        timeout=OLLAMA_TIMEOUT
    )

    response.raise_for_status()

    data = response.json()

    return data["response"]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("INSURANCE RAG ASSISTANT")
    print("=" * 80)

    print("\nType 'exit' to quit.")

    while True:

        question = input("\nQuestion: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            break

        if not question:
            continue

        print("\nSearching policy documents...")

        try:

            results = search_knowledge_base(
                question,
                top_k=TOP_K
            )

        except Exception as e:

            print("\nERROR: Knowledge base search failed.")
            print(f"Details: {e}")
            continue

        if not results:

            print(
                "\nI could not find relevant information "
                "in the policy documents."
            )

            continue

        print("✓ Relevant policy sections found.")
        print(f"✓ Retrieved {len(results)} policy chunks.")

        try:

            print("\nGenerating answer...")

            answer = generate_answer(
                question,
                results
            )

            print("\n" + "=" * 80)
            print("ANSWER")
            print("=" * 80)

            print(answer)

            print("=" * 80)

        except requests.exceptions.Timeout:

            print("\nERROR: Ollama took too long to respond.")
            print("Try reducing TOP_K or MAX_CHARS_PER_CHUNK.")

        except requests.exceptions.ConnectionError:

            print("\nERROR: Could not connect to Ollama.")
            print("Make sure Ollama is running on localhost:11434.")

        except requests.RequestException as e:

            print("\nERROR: Ollama request failed.")
            print(f"Details: {e}")

        except Exception as e:

            print("\nERROR: Unexpected error.")
            print(f"Details: {e}")


if __name__ == "__main__":
    main()

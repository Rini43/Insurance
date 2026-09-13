from search_knowledge_base import search_knowledge_base


TEST_QUESTIONS = [
    "What is covered under own damage?",
    "What is the personal accident cover for owner driver?",
    "What are the general exclusions?",
    "What is zero depreciation?",
    "What is roadside assistance?",
    "What is the claim procedure?",
    "What documents are required for a claim?",
    "Is engine damage covered?",
    "What is the voluntary deductible?",
    "What happens in case of total loss?"
]


def main():

    print("=" * 80)
    print("INSURANCE RETRIEVAL EVALUATION")
    print("=" * 80)

    total = len(TEST_QUESTIONS)
    passed = 0

    for number, question in enumerate(
        TEST_QUESTIONS,
        1
    ):

        results = search_knowledge_base(
            question,
            top_k=5
        )

        print("\n" + "-" * 80)
        print(f"TEST {number}/{total}")
        print("-" * 80)

        print(f"Question: {question}")

        if results:

            passed += 1

            best = results[0]

            print("✓ Retrieval successful")
            print(f"Company : {best['company']}")
            print(f"Document: {best['document']}")
            print(
                f"Pages   : "
                f"{best['page_start']}-{best['page_end']}"
            )
            print(
                f"Distance: "
                f"{best['distance']:.4f}"
            )

        else:

            print("✗ No results")

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print(f"Tests passed : {passed}/{total}")

    if passed == total:
        print("RESULT: PASS")
    else:
        print("RESULT: CHECK REQUIRED")


if __name__ == "__main__":
    main()

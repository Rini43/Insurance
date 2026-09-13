import json
import re
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "processed" / "cleaned"
OUTPUT_DIR = BASE_DIR / "processed" / "chunks"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# ============================================================
# MOJIBAKE / ENCODING VALIDATION
# ============================================================

# These are characters commonly produced when UTF-8 text
# has been incorrectly decoded as Windows-1252/Latin-1.
MOJIBAKE_MARKERS = [
    "\u00e2",       # â
    "\u00ef",       # ï
    "\ufffd",       # � replacement character
]


def check_for_mojibake(text):
    """
    Check whether text contains obvious mojibake characters.

    Returns a list of markers found.
    """

    found = []

    for marker in MOJIBAKE_MARKERS:
        if marker in text:
            found.append(marker)

    return found


def validate_unicode(text, filename):
    """
    Validate text before it is chunked.
    """

    bad_markers = check_for_mojibake(text)

    if bad_markers:
        print(
            f"  WARNING: Possible mojibake found in {filename}: "
            f"{bad_markers}"
        )
        return False

    # Check that important Unicode punctuation is preserved.
    unicode_chars = {
        "en dash": "\u2013",
        "em dash": "\u2014",
        "left quote": "\u201c",
        "right quote": "\u201d",
        "apostrophe": "\u2019",
    }

    found_unicode = []

    for name, char in unicode_chars.items():
        if char in text:
            found_unicode.append(name)

    if found_unicode:
        print(
            "  ✓ Unicode punctuation preserved: "
            + ", ".join(found_unicode)
        )
    else:
        print("  ✓ No suspicious mojibake detected.")

    return True


# ============================================================
# COMPANY NAME
# ============================================================

def clean_company_name(filename):
    """
    Convert the source filename into a readable company name.
    """

    name = Path(filename).stem

    replacements = {
        "Acko Car Package Policy":
            "Acko General Insurance",

        "Bajaj Allianz General Insurance Company":
            "Bajaj Allianz General Insurance",

        "Cholamandalam MS General Insurance Company Limited":
            "Cholamandalam MS General Insurance",

        "Digit Private Car Policy":
            "Digit Insurance",

        "HDFC ERGO Comprehensive":
            "HDFC ERGO",

        "ICICI Lombard General Insurance Company Limited":
            "ICICI Lombard General Insurance",

        "IndusInd General Insurance Company Limited":
            "IndusInd General Insurance",

        "SBI General Insurance Company Limited":
            "SBI General Insurance",

        "Shriram General Insurance Co. Ltd":
            "Shriram General Insurance",

        "TATA AIG General Insurance":
            "TATA AIG General Insurance",
    }

    return replacements.get(name, name)


# ============================================================
# PAGE EXTRACTION
# ============================================================

def extract_pages(text):
    """
    Split cleaned document text into individual pages.

    Expected format:

    ===== PAGE 1 =====
    page content...

    ===== PAGE 2 =====
    page content...
    """

    pattern = r"===== PAGE (\d+) ====="

    matches = list(re.finditer(pattern, text))

    pages = []

    for i, match in enumerate(matches):

        page_number = int(match.group(1))

        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        content = text[start:end].strip()

        if content:
            pages.append({
                "page_number": page_number,
                "content": content
            })

    return pages


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize whitespace while preserving Unicode characters.
    """

    # Normalize Windows-style line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Normalize tabs/spaces.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_into_words(text):
    """
    Convert text into a list of words.
    """

    return text.split()


# ============================================================
# CHUNK CREATION
# ============================================================

def create_chunks(pages, document_name, company_name):
    """
    Create overlapping word-based chunks while preserving
    page metadata.
    """

    words_with_pages = []

    for page in pages:

        page_number = page["page_number"]

        content = normalize_text(page["content"])

        words = split_into_words(content)

        for word in words:
            words_with_pages.append({
                "word": word,
                "page": page_number
            })

    chunks = []

    total_words = len(words_with_pages)

    start = 0
    chunk_number = 1

    while start < total_words:

        end = min(start + CHUNK_SIZE, total_words)

        selected = words_with_pages[start:end]

        if not selected:
            break

        chunk_words = [
            item["word"]
            for item in selected
        ]

        page_numbers = sorted(
            set(
                item["page"]
                for item in selected
            )
        )

        chunk_text = " ".join(chunk_words).strip()

        # Validate the actual chunk before saving it.
        bad_markers = check_for_mojibake(chunk_text)

        if bad_markers:
            raise ValueError(
                f"Mojibake detected while creating chunk "
                f"{chunk_number} of {document_name}: "
                f"{bad_markers}"
            )

        chunk_id = (
            f"{Path(document_name).stem.lower().replace(' ', '_')}"
            f"_chunk_{chunk_number:04d}"
        )

        chunk = {
            "chunk_id": chunk_id,
            "document": document_name,
            "company": company_name,
            "page_start": page_numbers[0],
            "page_end": page_numbers[-1],
            "pages": page_numbers,
            "chunk_number": chunk_number,
            "word_count": len(chunk_words),
            "content": chunk_text
        }

        chunks.append(chunk)

        chunk_number += 1

        if end >= total_words:
            break

        start = end - CHUNK_OVERLAP

    return chunks


# ============================================================
# JSON VALIDATION
# ============================================================

def validate_chunks(chunks, document_name):
    """
    Validate chunks before writing them to JSON.
    """

    for chunk in chunks:

        content = chunk.get("content", "")

        # Replacement character must never exist.
        if "\ufffd" in content:
            raise ValueError(
                f"Replacement character found in "
                f"{document_name}, chunk {chunk['chunk_number']}"
            )

        # Obvious mojibake must never exist.
        for marker in ["\u00e2", "\u00ef"]:
            if marker in content:
                raise ValueError(
                    f"Mojibake character {repr(marker)} found in "
                    f"{document_name}, chunk {chunk['chunk_number']}"
                )

    return True


# ============================================================
# SAVE JSON
# ============================================================

def save_json(output_file, data):
    """
    Save JSON explicitly as UTF-8.

    ensure_ascii=False is important because it keeps Unicode
    characters such as – — “ ” ’ as actual Unicode characters.
    """

    with output_file.open(
        "w",
        encoding="utf-8",
        newline="\n"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )

        f.write("\n")


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    text_files = sorted(
        INPUT_DIR.glob("*.txt")
    )

    if not text_files:

        print(
            f"No .txt files found in: "
            f"{INPUT_DIR}"
        )

        return

    print(
        f"Found {len(text_files)} cleaned text files."
    )

    print()

    all_chunks = []

    processed_documents = 0

    for text_file in text_files:

        print("=" * 80)

        print(
            f"Processing: {text_file.name}"
        )

        # ----------------------------------------------------
        # READ CLEANED TEXT AS UTF-8
        # ----------------------------------------------------

        text = text_file.read_text(
            encoding="utf-8"
        )

        print(
            f"Characters read: {len(text):,}"
        )

        # ----------------------------------------------------
        # VALIDATE INPUT
        # ----------------------------------------------------

        if not validate_unicode(
            text,
            text_file.name
        ):
            raise ValueError(
                f"Input text failed Unicode validation: "
                f"{text_file.name}"
            )

        # ----------------------------------------------------
        # EXTRACT PAGES
        # ----------------------------------------------------

        pages = extract_pages(text)

        if not pages:

            print(
                "WARNING: No page markers found."
            )

            continue

        # ----------------------------------------------------
        # DOCUMENT METADATA
        # ----------------------------------------------------

        document_name = text_file.stem

        company_name = clean_company_name(
            text_file.name
        )

        # ----------------------------------------------------
        # CREATE CHUNKS
        # ----------------------------------------------------

        chunks = create_chunks(
            pages=pages,
            document_name=document_name,
            company_name=company_name
        )

        # ----------------------------------------------------
        # VALIDATE CHUNKS
        # ----------------------------------------------------

        validate_chunks(
            chunks,
            document_name
        )

        # ----------------------------------------------------
        # SAVE INDIVIDUAL DOCUMENT JSON
        # ----------------------------------------------------

        output_file = (
            OUTPUT_DIR /
            f"{document_name}.json"
        )

        save_json(
            output_file,
            chunks
        )

        # ----------------------------------------------------
        # RE-READ JSON TO VERIFY UTF-8
        # ----------------------------------------------------

        verification_text = output_file.read_text(
            encoding="utf-8"
        )

        verification_data = json.loads(
            verification_text
        )

        validate_chunks(
            verification_data,
            document_name
        )

        # ----------------------------------------------------
        # ADD TO COMBINED LIST
        # ----------------------------------------------------

        all_chunks.extend(
            verification_data
        )

        processed_documents += 1

        print(
            f"Pages: {len(pages)}"
        )

        print(
            f"Chunks: {len(chunks)}"
        )

        print(
            f"Output: {output_file}"
        )

        print(
            "✓ JSON UTF-8 verification passed."
        )

    # ========================================================
    # SAVE COMBINED JSON
    # ========================================================

    combined_file = (
        OUTPUT_DIR /
        "all_chunks.json"
    )

    save_json(
        combined_file,
        all_chunks
    )

    # --------------------------------------------------------
    # Verify combined file
    # --------------------------------------------------------

    combined_text = combined_file.read_text(
        encoding="utf-8"
    )

    combined_data = json.loads(
        combined_text
    )

    validate_chunks(
        combined_data,
        "all_chunks.json"
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()

    print("=" * 80)

    print(
        "CHUNKING COMPLETE"
    )

    print("=" * 80)

    print(
        f"Documents processed: "
        f"{processed_documents}"
    )

    print(
        f"Total chunks: "
        f"{len(all_chunks)}"
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Combined file: "
        f"{combined_file}"
    )

    print()

    print(
        "✓ UTF-8 input validation passed"
    )

    print(
        "✓ Chunk validation passed"
    )

    print(
        "✓ JSON UTF-8 validation passed"
    )

    print(
        "✓ No obvious mojibake detected"
    )

    print()

    # --------------------------------------------------------
    # Show an actual Unicode example
    # --------------------------------------------------------

    for chunk in all_chunks:

        content = chunk.get(
            "content",
            ""
        )

        if "\u2013" in content:

            index = content.find(
                "\u2013"
            )

            print(
                "Unicode verification example:"
            )

            print(
                repr(
                    content[
                        max(0, index - 20):
                        index + 30
                    ]
                )
            )

            print(
                "Unicode code point:",
                hex(
                    ord(
                        content[index]
                    )
                )
            )

            break

    print()

    print(
        "RESULT: PASS"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

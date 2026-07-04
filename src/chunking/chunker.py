"""
Chunker
Splits extracted page text into overlapping token-based chunks,
preserving metadata (pdf_id, filename, page_number) for each chunk.
"""

import json
from pathlib import Path

import tiktoken
from tqdm import tqdm

INPUT_FILE = Path("data/processed/extracted_pages.json")
OUTPUT_FILE = Path("data/processed/chunks.json")

CHUNK_SIZE_TOKENS = 700     # target chunk size (within the 500-1000 range asked for)
CHUNK_OVERLAP_TOKENS = 140  # 20% overlap of 700

# Using a common tokenizer just to count/split tokens consistently.
# We are not calling any OpenAI API here — this is fully local and free.
ENCODING = tiktoken.get_encoding("cl100k_base")


def chunk_text_by_tokens(text: str, chunk_size: int, overlap: int):
    """
    Splits `text` into a list of overlapping chunks, each around `chunk_size` tokens,
    with `overlap` tokens repeated between consecutive chunks (so context isn't lost
    at chunk boundaries).
    """
    if not text.strip():
        return []

    tokens = ENCODING.encode(text)
    chunks = []

    start = 0
    total_tokens = len(tokens)

    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        chunk_token_slice = tokens[start:end]
        chunk_text = ENCODING.decode(chunk_token_slice)
        chunks.append(chunk_text.strip())

        if end == total_tokens:
            break

        # move start forward, but re-include the last `overlap` tokens
        start = end - overlap

    return chunks


def main():
    if not INPUT_FILE.exists():
        print(f"Could not find {INPUT_FILE}. Run pdf_extractor.py first.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        pages = json.load(f)

    all_chunks = []
    chunk_counter = 0

    for page in tqdm(pages, desc="Chunking pages"):
        page_chunks = chunk_text_by_tokens(
            page["text"], CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS
        )

        for i, chunk_text in enumerate(page_chunks):
            chunk_counter += 1
            all_chunks.append({
                "chunk_id": f"{page['pdf_id']}_p{page['page_number']}_c{i}",
                "pdf_id": page["pdf_id"],
                "filename": page["filename"],
                "page_number": page["page_number"],
                "chunk_index_in_page": i,
                "text": chunk_text,
                "language": page.get("language", "unknown"),
                "source": page.get("source", "native"),
                "token_count": len(ENCODING.encode(chunk_text)),
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Created {len(all_chunks)} chunks from {len(pages)} pages -> saved to {OUTPUT_FILE}")

    # quick sanity stats
    token_counts = [c["token_count"] for c in all_chunks]
    if token_counts:
        print(f"Chunk token count: min={min(token_counts)}, max={max(token_counts)}, avg={sum(token_counts)//len(token_counts)}")


if __name__ == "__main__":
    main()
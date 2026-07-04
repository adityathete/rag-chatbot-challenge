"""
Embedding + Vector Store
Embeds text chunks using a free, open-source embedding model (BAAI/bge-small-en-v1.5)
and stores them in a local ChromaDB vector database, along with their metadata.
"""

import json
from pathlib import Path

import torch
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHUNKS_FILE = Path("data/processed/chunks.json")
VECTOR_DB_DIR = Path("vector_store")
COLLECTION_NAME = "pdf_chunks"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 32  # how many chunks to embed at once — bigger uses more GPU memory


def get_device():
    """Use the GPU if available, otherwise fall back to CPU."""
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("No GPU found, using CPU (will be slower).")
    return "cpu"


def load_chunks():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Could not find {CHUNKS_FILE}. Run chunker.py first.")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    device = get_device()

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} (first run will download it)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks to embed.")

    # Set up ChromaDB — this saves to disk in VECTOR_DB_DIR, persists across runs
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

    # Start fresh each time we run this script, to avoid duplicate entries
    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(name=COLLECTION_NAME)

    texts = [c["text"] for c in chunks]

    print("Embedding chunks in batches...")
    all_embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # important: makes cosine similarity search work correctly
    )

    print("Storing embeddings + metadata in ChromaDB...")
    # ChromaDB wants metadata values to be str/int/float/bool only (no None, no nested dicts)
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Adding to ChromaDB"):
        batch_chunks = chunks[i:i + BATCH_SIZE]
        batch_embeddings = all_embeddings[i:i + BATCH_SIZE]

        collection.add(
            ids=[c["chunk_id"] for c in batch_chunks],
            embeddings=[emb.tolist() for emb in batch_embeddings],
            documents=[c["text"] for c in batch_chunks],
            metadatas=[
                {
                    "pdf_id": c["pdf_id"],
                    "filename": c["filename"],
                    "page_number": c["page_number"],
                    "language": c["language"],
                    "source": c["source"],
                }
                for c in batch_chunks
            ],
        )

    print(f"\nDone. Stored {len(chunks)} chunk embeddings in ChromaDB at {VECTOR_DB_DIR}/")


if __name__ == "__main__":
    main()
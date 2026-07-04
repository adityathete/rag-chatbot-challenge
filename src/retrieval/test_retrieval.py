"""
Quick manual test: given a query, embed it and search ChromaDB
for the most relevant chunks. This lets us sanity-check retrieval
quality before wiring up the full RAG pipeline with an LLM.
"""

print("Loading libraries (this can take 30-60 seconds on first run, please wait)...")
import torch
print("  torch loaded.")
import chromadb
print("  chromadb loaded.")
from sentence_transformers import SentenceTransformer
print("  sentence-transformers loaded.")

VECTOR_DB_DIR = "vector_store"
COLLECTION_NAME = "pdf_chunks"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 5


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    print(f"Collection has {collection.count()} chunks stored.\n")

    while True:
        query = input("Enter a question (or 'quit' to exit): ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
        )

        print(f"\nTop {TOP_K} results for: \"{query}\"\n")
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            text_preview = results["documents"][0][i][:200].replace("\n", " ")

            print(f"  [{i+1}] {meta['filename']} — page {meta['page_number']} (distance: {distance:.4f})")
            print(f"      \"{text_preview}...\"\n")

        print("-" * 80)


if __name__ == "__main__":
    main()
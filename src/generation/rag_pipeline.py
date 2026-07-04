"""
RAG Pipeline
Full pipeline: takes a user question, retrieves relevant chunks from ChromaDB,
sends them to a local LLM (via Ollama) to generate an answer, and returns
the answer along with source citations (PDF filename + page number).
"""

import time

import chromadb
import ollama
import torch
from sentence_transformers import SentenceTransformer

VECTOR_DB_DIR = "vector_store"
COLLECTION_NAME = "pdf_chunks"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "qwen2.5:3b"
TOP_K = 2

SYSTEM_PROMPT = """Answer using ONLY the given context. If the answer isn't in the context, 
say "I couldn't find this information in the documents." Be concise. Do not add your own citations."""


class RAGPipeline:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model on {device}...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

        print("Connecting to vector database...")
        client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        self.collection = client.get_collection(COLLECTION_NAME)
        print(f"Ready. Collection has {self.collection.count()} chunks.\n")

    def retrieve(self, query: str, top_k: int = TOP_K):
        """Embed the query and fetch the top-k most relevant chunks."""
        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True
        )[0].tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "filename": results["metadatas"][0][i]["filename"],
                "page_number": results["metadatas"][0][i]["page_number"],
                "distance": results["distances"][0][i],
            })
        return chunks

    def build_context(self, chunks, max_chars_per_chunk=1200):
        """Format retrieved chunks into a context block for the LLM prompt."""
        context_parts = []
        for i, chunk in enumerate(chunks):
            truncated_text = chunk["text"][:max_chars_per_chunk]
            context_parts.append(
                f"[Passage {i+1}] (Source: {chunk['filename']}, page {chunk['page_number']})\n{truncated_text}"
            )
        return "\n\n".join(context_parts)

    def generate_answer(self, query: str, chunks):
        """Send the question + retrieved context to the LLM and get an answer."""
        context = self.build_context(chunks)

        user_prompt = f"""Context passages:
{context}

Question: {query}

Answer based only on the context above:"""

        response = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "num_predict": 100,
                "num_ctx": 1024,
                "temperature": 0.1,   # low randomness = more consistent, factual answers
            },
            keep_alive="30m",
        )
        return response["message"]["content"]

    def answer_question(self, query: str):
        """Full pipeline: retrieve -> generate -> return answer + sources."""
        start_time = time.time()

        chunks = self.retrieve(query)
        retrieval_time = time.time() - start_time

        answer = self.generate_answer(query, chunks)
        total_time = time.time() - start_time

        # build a de-duplicated source list (same page might appear once, that's fine;
        # same filename+page combo shouldn't repeat)
        seen = set()
        sources = []
        for chunk in chunks:
            key = (chunk["filename"], chunk["page_number"])
            if key not in seen:
                seen.add(key)
                sources.append(key)

        return {
            "answer": answer,
            "sources": sources,
            "retrieval_time": retrieval_time,
            "total_time": total_time,
        }


def main():
    pipeline = RAGPipeline()

    while True:
        query = input("Ask a question (or 'quit' to exit): ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        result = pipeline.answer_question(query)

        print(f"\nAnswer:\n{result['answer']}\n")
        print("Sources:")
        for filename, page in result["sources"]:
            print(f"  - {filename}, page {page}")
        print(f"\n(Retrieval: {result['retrieval_time']:.2f}s | Total: {result['total_time']:.2f}s)")
        print("-" * 80)


if __name__ == "__main__":
    main()
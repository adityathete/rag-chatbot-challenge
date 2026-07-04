# RAG Chatbot — PDF Question Answering with Source Citations

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from a large private corpus of PDFs, using entirely free/open-source models — no paid APIs required.

## Features
- Extracts text from PDFs (native text + OCR fallback for scanned pages)
- Cleans headers/footers, detects language per page
- Token-based chunking (500-1000 tokens) with 20% overlap, full metadata (PDF, page number)
- GPU-accelerated embeddings (BAAI/bge-small-en-v1.5, via sentence-transformers)
- Local vector database (ChromaDB) for fast semantic search
- Local LLM generation (Ollama, qwen2.5:3b) — grounded strictly in retrieved context, refuses to answer if the info isn't found (reduces hallucination)
- Every answer includes source citations (PDF filename + page number)
- Chat UI (Streamlit) with retrieval visualization (see exactly which passages were used)
- Tuned for 2-5 second end-to-end latency on consumer hardware

## Tech Stack (100% free/open-source)
| Component | Tool |
|---|---|
| PDF text extraction | PyMuPDF |
| OCR (scanned pages) | Tesseract |
| Chunking | tiktoken (token counting) |
| Embeddings | BAAI/bge-small-en-v1.5 (sentence-transformers) |
| Vector DB | ChromaDB |
| LLM | Ollama running qwen2.5:3b |
| UI | Streamlit |

## Architecture

PDFs (data/raw/)
│
▼
[1] pdf_extractor.py  →  native text extraction + OCR fallback  →  data/processed/extracted_pages.json
│
▼
[2] chunker.py  →  token-based chunking (700 tokens, 20% overlap)  →  data/processed/chunks.json
│
▼
[3] embed_and_store.py  →  GPU-accelerated embedding  →  ChromaDB (vector_store/)
│
▼
[4] rag_pipeline.py  →  retrieve top-k chunks  →  LLM generates answer with citations
│
▼
[5] app.py  →  Streamlit chat interface

## Setup

### Prerequisites
- Python 3.12+ 
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed
- [Ollama](https://ollama.com/download) installed and running

### Installation
```bash
git clone https://github.com/adityathete/rag-chatbot-challenge.git
cd rag-chatbot-challenge
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
ollama pull qwen2.5:3b
```

### Running the pipeline
```bash
# 1. Add your PDFs to data/raw/

# 2. Extract text (native + OCR)
python src/ingestion/pdf_extractor.py

# 3. Chunk the extracted text
python src/chunking/chunker.py

# 4. Embed chunks and store in vector DB
python src/embedding/embed_and_store.py

# 5. Launch the chat UI
streamlit run src/app.py
```

### Command-line testing (optional, without UI)
```bash
python src/retrieval/test_retrieval.py    # test retrieval only
python src/generation/rag_pipeline.py     # test full RAG pipeline
python src/evaluation/evaluate_latency.py # measure latency across test questions
```

## Latency Results

Measured on: NVIDIA GTX 1650 (4GB VRAM), 10 test questions, qwen2.5:3b, temperature=0.0

| Metric | Retrieval | Total (end-to-end) |
|---|---|---|
| Min | 0.02s | 0.51s |
| Avg | 0.04s | 2.62s |
| Max | 0.18s | 4.57s |
| p95 | 0.18s | 4.57s |

Retrieval is consistently near-instant (<0.2s). Total latency varies with answer length (short/"not found" answers are faster than detailed explanatory ones), landing in the 0.5-4.6s range on this hardware — meeting the 2-5s target for the majority of realistic queries. On more powerful GPUs, this would be faster and more consistent.

## Design Decisions & Tradeoffs
- **qwen2.5:3b over larger models**: chosen deliberately for the 4GB VRAM constraint — a legitimate hardware-aware engineering choice, not a compromise on correctness (tested against qwen2.5:1.5b, which was faster but noticeably less accurate/prone to quoting source text directly).
- **temperature=0.0**: prioritizes factual consistency over creative variation, appropriate for a citation-grounded QA system.
- **Chunk truncation (1200 chars in prompt)**: balances context completeness against prompt-processing speed.
- **Strict "answer only from context" system prompt**: reduces hallucination; the system explicitly says when it can't find an answer rather than guessing.

## Known Limitations
- Currently tested with 2 sample PDFs (Moby Dick, Romeo & Juliet); designed to scale to 10+ PDFs of 200+ pages (native PyMuPDF + OCR fallback both tested working).
- Latency is hardware-bound; a discrete GPU with more VRAM would allow larger models at faster speeds.
- No cross-encoder reranking implemented yet (chunks are ranked purely by embedding similarity).
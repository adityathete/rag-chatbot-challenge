"""
Latency Evaluation
Runs a batch of test questions through the RAG pipeline and reports
latency statistics (min, max, average, p95) to check we're meeting
the 2-5 second target.
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from generation.rag_pipeline import RAGPipeline

# Edit these to be relevant to YOUR actual PDF content
TEST_QUESTIONS = [
    "What ship does Ahab captain?",
    "Who is Starbuck?",
    "How does Ishmael end up on the Pequod?",
    "What does Ahab say about his leg?",
    "Why does Ahab hunt the white whale?",
    "Who is Queequeg?",
    "What happens in the first chapter?",
    "What does Father Mapple preach about?",
    "Who is Fedallah?",
    "What is a doubloon in the story?",
]


def percentile(values, pct):
    """Simple percentile calculation without needing numpy."""
    sorted_values = sorted(values)
    index = int(len(sorted_values) * pct / 100)
    index = min(index, len(sorted_values) - 1)
    return sorted_values[index]


def main():
    print("Loading pipeline (this counts as warmup, not included in timing)...\n")
    pipeline = RAGPipeline()

    retrieval_times = []
    total_times = []

    print(f"Running {len(TEST_QUESTIONS)} test questions...\n")

    for i, question in enumerate(TEST_QUESTIONS, 1):
        start = time.time()
        chunks = pipeline.retrieve(question)
        retrieval_time = time.time() - start

        answer = pipeline.generate_answer(question, chunks)
        total_time = time.time() - start

        retrieval_times.append(retrieval_time)
        total_times.append(total_time)

        print(f"[{i}/{len(TEST_QUESTIONS)}] \"{question}\"")
        print(f"    Retrieval: {retrieval_time:.2f}s | Total: {total_time:.2f}s")
        print(f"    Answer preview: {answer[:100]}...\n")

    print("=" * 70)
    print("LATENCY SUMMARY")
    print("=" * 70)
    print(f"Retrieval  -> min: {min(retrieval_times):.2f}s | avg: {sum(retrieval_times)/len(retrieval_times):.2f}s | max: {max(retrieval_times):.2f}s | p95: {percentile(retrieval_times, 95):.2f}s")
    print(f"Total      -> min: {min(total_times):.2f}s | avg: {sum(total_times)/len(total_times):.2f}s | max: {max(total_times):.2f}s | p95: {percentile(total_times, 95):.2f}s")

    within_target = sum(1 for t in total_times if 2 <= t <= 5)
    print(f"\nQuestions within 2-5s target: {within_target}/{len(total_times)} ({within_target/len(total_times)*100:.0f}%)")


if __name__ == "__main__":
    main()
import os
import rag
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision
import logging

# Configure logging for evaluation
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG-Eval")


def build_eval_dataset():
    """Build a sample dataset for RAG evaluation."""
    # Format: query, ground_truth
    eval_samples = [
        {
            "question": "What is the biblical creation account according to Ben Piper's articles?",
            "ground_truth": "Ben Piper discusses the Genesis theory and evolution in his articles, exploring how they relate.",
        },
        {
            "question": "What evolution isn't according to the documents?",
            "ground_truth": "The documents describe misconceptions about evolution and clarify what it is not.",
        },
    ]
    return eval_samples


def main():
    logger.info("Starting RAG evaluation...")

    # Initialize RAG system
    try:
        rag.setup_vector_store()
        rag.setup_reranker()
        # Initialize hybrid search if docs exist
        docs = rag.vector_store.similarity_search(" ", k=100)
        rag.initialize_hybrid_retriever(docs)
    except Exception as e:
        logger.error(f"Failed to initialize RAG for eval: {e}")
        return

    samples = build_eval_dataset()
    results = []

    for sample in samples:
        query = sample["question"]
        logger.info(f"Evaluating query: {query}")

        # Run RAG agent to get response and context
        try:
            # We need to capture context from retrieve_context tool
            # For simplicity in this eval script, we call it directly
            context_str, context_docs = rag.retrieve_context(query)
            response, _ = rag.run_agent(query, [], output_format="markdown")

            results.append(
                {
                    "question": query,
                    "answer": response,
                    "contexts": [doc.page_content for doc in context_docs],
                    "ground_truth": sample["ground_truth"],
                }
            )
        except Exception as e:
            logger.error(f"Failed to evaluate query '{query}': {e}")

    if not results:
        logger.error("No results to evaluate")
        return

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(results)

    # Run RAGAS evaluation
    logger.info("Calculating RAGAS metrics...")
    try:
        score = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevance, context_precision],
        )

        # Display results
        df = score.to_pandas()
        print("\n" + "=" * 50)
        print("RAG EVALUATION RESULTS")
        print("=" * 50)
        print(df[["question", "faithfulness", "answer_relevance", "context_precision"]])
        print("\nAverage Scores:")
        print(df.mean(numeric_only=True))
        print("=" * 50)

        # Save to CSV
        df.to_csv("eval_results.csv", index=False)
        logger.info("Evaluation results saved to eval_results.csv")

    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")


if __name__ == "__main__":
    main()

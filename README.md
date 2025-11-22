# LangChain Q&A Application
This repository contains a simple question-answering (Q&A) application built using the LangChain library. The application is designed to retrieve information from a collection of documents and provide answers to user queries.

## Features
- Loads documents from a list of URLs, extracting only post titles, headers, and content.
- Splits the loaded documents into smaller chunks for efficient processing.
- Embeds each document chunk using OpenAI's text-embedding model.
- Stores the embedded documents in an InMemoryVectorStore for efficient retrieval.
- Provides a custom tool to retrieve context from the stored documents based on user queries.
- Implements two different methods for generating responses:
  1. A Retrieval-only Agent (RAG) that uses the custom tool to fetch relevant documents and generates an answer based on those documents.
  2. A Retrieval with RAG Chains agent that injects context into the state messages before generating a response.

## Configuring

Put your OpenAI API key in `.env` like so:

`OPENAI_API_KEY=sk-...`

Modify `rag.py` to change the chat model, embeddings model, and URLs you want to index.

## Running

`uv run --env-file .env -- rag.py`

Enter your query when prompted. e.g. "What's the difference between whey and collagen proteins?"


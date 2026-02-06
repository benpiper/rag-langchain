# LangChain RAG Q&A Application

This repository contains a robust question-answering (Q&A) application built using the LangChain library. The application retrieves information from web URLs and local documents to provide accurate, cited answers to user queries.

## 🚀 Recent Enhancements

- **Hybrid Search**: Combines BM25 (keyword) and Vector search using Reciprocal Rank Fusion (RRF).
- **Metadata-Aware Re-ranking**: Improved re-ranking that considers document titles/filenames, ensuring better retrieval for author-specific queries.
- **Query Expansion**: Automatically generates multiple query variations to improve retrieval recall.
- **Streamlit Web UI**: A modern, interactive web interface for a premium chat experience.
- **Improved CLI Commands**: Added `/sources` to quickly list all indexed documents.
- **Multi-Source Loading**: Native support for PDF, DOCX, and CSV files in addition to Text and Markdown.
- **Recursive Loading**: Support for recursive directory searching in the `docs/` folder.

## Features

### Core Retrieval Pipeline
- **Hybrid Retrieval**: Pairs traditional keyword matching (BM25) with semantic vector search.
- **Intelligent Re-ranking**: Uses a Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`) to score document relevance, now with metadata awareness.
- **Multi-Query Expansion**: Uses the LLM to understand different facets of the user's intent.
- **Data Persistence**: Uses **Milvus Lite** to store document embeddings locally (`milvus_demo.db`).

### Multi-Source Indexing
- **Web URLs**: Loads all URLs listed in `sources.txt`.
- **Local Documents**: Supports `.txt`, `.md`, `.pdf`, `.docx`, and `.csv` files in the `docs/` directory, with recursive searching enabled.

### User Experience
- **Streamlit Web UI**: Modern glassmorphism UI with configuration sidebar.
- **Interactive CLI**: Multi-turn conversations with history and source management.
- **Automated Summarization**: Transparently manages long conversation context.

---

## 🖥️ User Interfaces

### 1. Web UI (Streamlit)
For a premium, interactive experience, use the Streamlit interface:
```bash
uv run streamlit run app.py
```

### 2. Interactive CLI
For terminal-based usage:
```bash
uv run --env-file .env -- rag.py --interactive
```

**Interactive CLI Commands:**
- `/sources` - List all unique files currently in the index.
- `/history` - Display conversation history.
- `/save`    - Save current conversation to a JSON file.
- `/load`    - Load a previous conversation.
- `/clear`   - Clear current history.
- `/exit`    - Exit the application.

---

## Configuration

### Quick Start
1. **API Key**: Add your OpenAI API key to `.env`: `OPENAI_API_KEY=sk-...`
2. **Sources**: Add URLs to `sources.txt` or files to `docs/`.
3. **Run**: `uv run --env-file .env -- rag.py`

### Advanced Configuration (`config.yaml`)
Key areas in `config.yaml`:
- **retrieval**: Adjust `k` and hybrid search weights.
- **document_processing**: Add new file patterns to `supported_formats` and ensure `local_docs_dir` is correct.
- **prompts**: Customize the agent's personality and instructions.

---

## Project Structure

```
rag-langchain/
├── rag.py                      # Core RAG logic & CLI
├── app.py                      # Streamlit Web Interface
├── evaluate.py                 # RAGAS evaluation script
├── config.yaml                 # Main configuration
├── sources.txt                 # Web URLs to index
├── docs/                       # Local documents (PDF, DOCX, CSV, MD, TXT)
├── conversations/              # Saved chat histories
└── milvus_demo.db              # Vector database
```

## 📊 Evaluation

The project includes an evaluation framework using **RAGAS** to measure faithfulness, answer relevance, and context precision.

To run the evaluation:
```bash
uv run --env-file .env -- evaluate.py
```
Results will be saved to `evaluation_results.csv`, providing a quantitative baseline for your RAG performance.

## Troubleshooting

**Q: Installation fails with `onnxruntime`?**
A: Ensure you are using the provided `pyproject.toml` which pins `onnxruntime<1.24.0` for Python 3.10 compatibility.

**Q: Retrieval isn't finding new files?**
A: Run with `--force-refresh` to rebuild the index (vector store + BM25). This is required any time you add new local documents or URLs.
```bash
uv run --env-file .env -- rag.py --force-refresh
```



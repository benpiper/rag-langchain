# LangChain RAG Q&A Application

This repository contains a robust question-answering (Q&A) application built using the LangChain library. The application retrieves information from web URLs and local documents to provide accurate, cited answers to user queries.

## Features

- **Data Persistence**: Uses **Milvus Lite** to store document embeddings locally (`milvus_demo.db`), avoiding redundant indexing.
- **Multi-Source Indexing**:
    - **Web**: Loads URLs defined in `sources.txt`.
    - **Local**: Indexes `.txt` and `.md` files from the `docs/` directory.
- **Source Attribution**: Responses explicitly cite the source (Title/URL) of the information.
- **Flexible Retrieval Modes**:
    1. **Agent Mode (Default)**: An AI agent that decides when to search the knowledge base. Best for general conversation.
    2. **Chain Mode**: Forces a search for every query. Best for strict Q&A.
- **Command Line Interface**: Options for non-interactive use, mode selection, and index management.

## Configuring

1. **API Key**: Put your OpenAI API key in `.env`:
   ```bash
   OPENAI_API_KEY=sk-...
   ```

2. **Web Sources**: Add URLs to `sources.txt` (one per line).

3. **Local Documents**: Place text or markdown files in the `docs/` directory.

## Running

### Basic Usage
Run the application in interactive mode (Agent default):
```bash
uv run --env-file .env -- rag.py
```

### Command Line Options

| Argument | Description | Example |
| :--- | :--- | :--- |
| `--mode` | Choose retrieval mode (`agent` or `chain`). | `--mode chain` |
| `--query` | Run a single query and exit. | `--query "What is overtraining?"` |
| `--force-refresh` | Delete the database and re-index all sources. | `--force-refresh` |

### Examples

**Run a single query in Chain mode:**
```bash
uv run --env-file .env -- rag.py --mode chain --query "Best exercises for back?"
```

**Update the index after changing sources:**
```bash
uv run --env-file .env -- rag.py --force-refresh
```



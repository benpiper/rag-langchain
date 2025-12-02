# LangChain RAG Q&A Application

This repository contains a robust question-answering (Q&A) application built using the LangChain library. The application retrieves information from web URLs and local documents to provide accurate, cited answers to user queries.

## Features

- **Data Persistence**: Uses **Milvus Lite** to store document embeddings locally (`milvus_demo.db`), avoiding redundant indexing.
- **Multi-Source Indexing**: **Both sources are indexed together**:
    - **Web URLs**: Loads all URLs listed in `sources.txt` (one per line).
    - **Local Documents**: Indexes all `.txt` and `.md` files from the `docs/` directory.
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
| `--query` | Run a single query and exit. | `--query "What is evolution?"` |
| `--force-refresh` | Delete the database and re-index all sources. | `--force-refresh` |

### Examples

**Run a single query in Chain mode:**
```bash
uv run --env-file .env -- rag.py --mode chain --query "Explain common descent"
```

**Update the index after changing sources:**
```bash
uv run --env-file .env -- rag.py --force-refresh
```

## How Indexing Works

The application **indexes both web URLs and local documents together** into a single vector store:

1. **First Run**: If `milvus_demo.db` doesn't exist or is empty, the application automatically indexes:
   - All URLs from `sources.txt`
   - All `.txt` and `.md` files from `docs/`

2. **Subsequent Runs**: The application reuses the existing index for faster startup.

3. **When to Re-index**: Use `--force-refresh` when:
   - You add/remove URLs in `sources.txt`
   - You add/update files in `docs/`
   - Web content has changed and you want fresh data

## Troubleshooting

**Q: URLs aren't being referenced in responses?**

A: You need to re-index after changing `sources.txt`:
```bash
uv run --env-file .env -- rag.py --force-refresh
```

**Q: Getting schema or metadata errors?**

A: Delete the database and let it rebuild:
```bash
rm -f milvus_demo.db
uv run --env-file .env -- rag.py
```



# LangChain RAG Q&A Application

This repository contains a robust question-answering (Q&A) application built using the LangChain library. The application retrieves information from web URLs and local documents to provide accurate, cited answers to user queries.

## Features

- **Data Persistence**: Uses **Milvus Lite** to store document embeddings locally (`milvus_demo.db`), avoiding redundant indexing.
- **Multi-Source Indexing**: **Both sources are indexed together**:
    - **Web URLs**: Loads all URLs listed in `sources.txt` (one per line).
    - **Local Documents**: Indexes all `.txt` and `.md` files from the `docs/` directory.
- **Source Attribution**: Responses explicitly cite the source (Title/URL) of the information.
- **Flexible Retrieval Modes** (Default: **Agent**):
    - **Agent Mode** ⭐ **(Default)**: 
        - The AI agent intelligently decides when to search the knowledge base
        - More conversational and natural for back-and-forth dialogue
        - Will skip retrieval for simple greetings, clarifications, or when general knowledge suffices
        - Best for: General conversation, mixed topics, follow-up questions
    - **Chain Mode**: 
        - Automatically retrieves from the knowledge base for every query
        - Guarantees that responses are grounded in your indexed documents
        - More deterministic and suitable for strict Q&A scenarios
        - Best for: Fact-checking, ensuring all answers cite sources, production Q&A systems
- **Command Line Interface**: Options for non-interactive use, mode selection, and index management.

## Retrieval Modes Explained

### Agent Mode (Default)

The agent has access to a `retrieve_context` tool but decides when to use it based on the query:

- **When it retrieves**: Complex questions about specific topics in your knowledge base
- **When it doesn't**: Greetings ("hello"), meta questions ("what can you do?"), or requests it can handle with general knowledge
- **Output style**: More conversational, may blend general knowledge with retrieved facts
- **Use case**: Interactive sessions where you want natural dialogue

### Chain Mode

Every query triggers a knowledge base search before responding:

- **Always retrieves**: Even for simple queries, the system searches your documents first
- **Output style**: Responses are strictly based on retrieved documents (or explicitly state when information isn't found)
- **Use case**: When you need to ensure every answer is grounded in your specific knowledge base

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



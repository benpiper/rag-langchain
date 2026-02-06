# LangChain RAG Q&A Application

This repository contains a robust question-answering (Q&A) application built using the LangChain library. The application retrieves information from web URLs and local documents to provide accurate, cited answers to user queries.

## Features

### Core Features
- **Configuration-Driven**: Centralized YAML configuration with support for local overrides
- **Data Persistence**: Uses **Milvus Lite** to store document embeddings locally (`milvus_demo.db`), avoiding redundant indexing
- **Multi-Source Indexing**: **Both sources are indexed together**:
    - **Web URLs**: Loads all URLs listed in `sources.txt` (one per line)
    - **Local Documents**: Indexes all `.txt` and `.md` files from the `docs/` directory
- **Multiple Embedding Providers**: Support for OpenAI and Ollama embeddings
- **Source Attribution**: Responses explicitly cite the source (Title/URL) of the information

### User Experience
- **Interactive Mode**: Multi-turn conversations with full context awareness
- **Conversation History**: Save, load, and resume conversations
- **Multiple Output Formats**: Plain text, Markdown, or JSON
- **Streaming Responses**: Real-time response streaming for better UX
- **Built-in Commands**: Help, history viewing, conversation management

### Retrieval Modes
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

## Configuration

### Quick Start Configuration

1. **API Key**: Put your OpenAI API key in `.env`:
   ```bash
   OPENAI_API_KEY=sk-...
   ```

2. **Web Sources**: Add URLs to `sources.txt` (one per line).

3. **Local Documents**: Place text or markdown files in the `docs/` directory.

### Advanced Configuration

The application now supports comprehensive configuration via YAML files:

#### Main Configuration File: `config.yaml`

All application settings are centralized in [config.yaml](config.yaml), including:

- **Logging**: Log level, format, and output file
- **LLM Settings**: Model name, temperature, max tokens
- **Embeddings**: Provider selection (OpenAI/Ollama) and model configuration
- **Vector Store**: Database path and Milvus settings
- **Document Processing**: Chunk size, overlap, supported file formats
- **Retrieval**: Number of documents to retrieve (k value)
- **System Prompts**: Customizable prompts for agent and chain modes

#### Local Configuration Overrides: `config.local.yaml`

Create a `config.local.yaml` file to override settings without modifying the main config:

```yaml
# Example: Use a different model
llm:
  model: gpt-4o
  temperature: 0.5

# Example: Retrieve more documents
retrieval:
  k: 10

# Example: Enable debug logging
logging:
  level: DEBUG
```

**Note**: `config.local.yaml` is git-ignored, so your personal settings won't be committed.

#### Configuration Precedence

Settings are applied in this order (later overrides earlier):
1. `config.yaml` (base configuration)
2. `config.local.yaml` (local overrides)
3. Command-line arguments (highest priority)

### Ollama Configuration

To use Ollama for embeddings instead of OpenAI:

**Option 1: Via config.local.yaml**
```yaml
embeddings:
  provider: ollama
  ollama:
    host: 192.168.88.86
    port: 11434
    model: embeddinggemma
```

**Option 2: Via command-line**
```bash
uv run --env-file .env -- rag.py --embedding-provider ollama --ollama-host 192.168.88.86 --ollama-model embeddinggemma
```

## Running

### Interactive Mode (Recommended)

Run the application in interactive mode with conversation history:
```bash
uv run --env-file .env -- rag.py --interactive
```

Or simply run without a query to start interactive mode:
```bash
uv run --env-file .env -- rag.py
```

**Interactive Mode Commands:**
- `/help` - Show available commands
- `/history` - Display conversation history
- `/save` - Save conversation to file
- `/load` - Load conversation from file
- `/clear` - Clear conversation history
- `/exit` - Exit interactive mode

**Features:**
- ✅ Multi-turn conversations with context
- ✅ Persistent conversation history
- ✅ Save/load conversations
- ✅ Real-time streaming responses

### Single Query Mode

Run a single query and exit:
```bash
uv run --env-file .env -- rag.py --query "Your question here"
```

### Command Line Options

| Argument | Description | Default (from config.yaml) |
| :--- | :--- | :--- |
| `--mode {agent,chain}` | Choose retrieval mode | `agent` |
| `--query QUERY` | Run a single query and exit | Interactive mode |
| `--interactive` | Explicitly start interactive mode | Auto if no query |
| `--output-format {plain,markdown,json}` | Output format | `plain` |
| `--load-conversation FILE` | Load conversation history from file | None |
| `--force-refresh` | Delete the database and re-index all sources | `false` |
| `--embedding-provider {openai,ollama}` | Embedding provider to use | `openai` |
| `--ollama-host HOST` | Ollama server host | `192.168.88.86` |
| `--ollama-model MODEL` | Ollama model name | `embeddinggemma` |

**Note**: Command-line arguments override settings in `config.yaml` and `config.local.yaml`.

### Examples

**Start interactive conversation:**
```bash
uv run --env-file .env -- rag.py --interactive
```

**Run a single query in Chain mode:**
```bash
uv run --env-file .env -- rag.py --mode chain --query "Explain common descent"
```

**Get JSON output:**
```bash
uv run --env-file .env -- rag.py --query "What is evolution?" --output-format json
```

**Continue a previous conversation:**
```bash
uv run --env-file .env -- rag.py --load-conversation conversations/conversation_20240101_120000.json --interactive
```

**Update the index after changing sources:**
```bash
uv run --env-file .env -- rag.py --force-refresh
```

## Interactive Mode & Conversations

### Multi-Turn Conversations

The interactive mode maintains conversation history, allowing the AI to reference previous questions and answers:

```
> What is natural selection?
[AI explains natural selection with citations]

> Can you give me an example?
[AI provides example, understanding "you" refers to the previous topic]

> How does this relate to evolution?
[AI connects the dots using conversation context]
```

### Conversation Management

**Save conversations for later:**
```
> /save
✓ Conversation saved to: conversations/conversation_20240315_143022.json
```

**Load previous conversations:**
```bash
uv run --env-file .env -- rag.py --load-conversation conversations/conversation_20240315_143022.json
```

Or within interactive mode:
```
> /load
Enter conversation file path: conversations/conversation_20240315_143022.json
✓ Loaded 10 messages
```

**View conversation history:**
```
> /history
Conversation history (10 messages):
1. [user]: What is natural selection?
2. [assistant]: Natural selection is the process...
...
```

### Output Formats

**Plain Text (Default):**
Human-readable output with formatted text.

**Markdown:**
```bash
uv run --env-file .env -- rag.py --output-format markdown --query "Explain evolution"
```

**JSON:**
Structured output for programmatic use:
```bash
uv run --env-file .env -- rag.py --output-format json --query "What is DNA?"
```

JSON output includes:
- Query text
- Response text
- Timestamp
- Full conversation history

## Common Configuration Scenarios

### Scenario 1: Adjust Retrieval Quality

Retrieve more documents for better context (may increase response time):
```yaml
# config.local.yaml
retrieval:
  k: 10  # Default is 6
```

### Scenario 2: Optimize for Speed

Use smaller chunks and retrieve fewer documents:
```yaml
# config.local.yaml
document_processing:
  chunk_size: 500
  chunk_overlap: 100

retrieval:
  k: 3
```

### Scenario 3: Better for Long Documents

Increase chunk size to preserve more context:
```yaml
# config.local.yaml
document_processing:
  chunk_size: 2000
  chunk_overlap: 400
```

### Scenario 4: Custom System Prompts

Modify how the AI responds by editing prompts in [config.yaml](config.yaml):
```yaml
prompts:
  agent_system_prompt: |
    You are a specialized assistant for biblical apologetics.
    Always cite sources and be respectful of different viewpoints.
    # ... rest of prompt
```

### Scenario 5: Use Different Models

Switch to a different LLM or embedding model:
```yaml
# config.local.yaml
llm:
  model: gpt-4o
  temperature: 0.3  # More deterministic

embeddings:
  openai:
    model: text-embedding-3-small  # Faster, cheaper
```

### Scenario 6: Debug Mode

Enable detailed logging for troubleshooting:
```yaml
# config.local.yaml
logging:
  level: DEBUG
```

Check the logs in `rag_app.log` for detailed execution traces.

## Project Structure

```
rag-langchain/
├── rag.py                      # Main application
├── config.yaml                 # Main configuration file
├── config.local.yaml.example   # Example local config overrides
├── config.local.yaml          # Your local overrides (git-ignored)
├── sources.txt                # Web URLs to index (one per line)
├── docs/                      # Local documents directory
│   ├── *.txt                 # Text files to index
│   └── *.md                  # Markdown files to index
├── conversations/            # Saved conversation history (git-ignored)
│   └── *.json               # Conversation files
├── milvus_demo.db            # Vector database (auto-created)
├── rag_app.log              # Application logs
├── .env                     # API keys (git-ignored)
└── pyproject.toml          # Python dependencies
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

**Q: Configuration not taking effect?**

A: Check the configuration precedence:
1. Verify your `config.local.yaml` syntax is valid YAML
2. Remember command-line args override config files
3. Check `rag_app.log` for config loading errors
4. Try running with `--help` to see current defaults

**Q: Where are my configuration files?**

A:
- `config.yaml` - Main configuration (committed to git)
- `config.local.yaml` - Your personal overrides (git-ignored, create if needed)
- `config.local.yaml.example` - Example override file for reference

**Q: Ollama connection errors?**

A: Verify Ollama settings:
```bash
# Test if Ollama is running
curl http://192.168.88.86:11434/api/tags

# Check your config
cat config.yaml | grep -A 5 "ollama:"
```

## Configuration Reference

All configurable settings in [config.yaml](config.yaml):

### Logging
- `level`: DEBUG, INFO, WARNING, ERROR, CRITICAL
- `log_file`: Path to log file
- `format`: Log message format

### LLM (Language Model)
- `model`: Model name (e.g., gpt-4.1, gpt-4o)
- `temperature`: Creativity/randomness (0.0-1.0)
- `max_tokens`: Maximum response length

### Embeddings
- `provider`: openai or ollama
- `openai.model`: OpenAI embedding model
- `ollama.host`: Ollama server host
- `ollama.port`: Ollama server port
- `ollama.model`: Ollama embedding model

### Vector Store
- `milvus.uri`: Database file path
- `milvus.auto_id`: Auto-generate document IDs

### Document Processing
- `chunk_size`: Characters per chunk
- `chunk_overlap`: Overlap between chunks
- `sources_file`: Path to URLs file
- `local_docs_dir`: Local documents directory
- `supported_formats`: File glob patterns to index

### Retrieval
- `k`: Number of documents to retrieve
- `test_k`: Documents for store existence check

### Prompts
- `agent_system_prompt`: Prompt for agent mode
- `chain_system_prompt`: Prompt for chain mode

### Defaults
- `mode`: agent or chain
- `embedding_provider`: openai or ollama
- `force_refresh`: true or false
- `output_format`: plain, markdown, or json
- `interactive`: true or false

### User Experience
- `conversations_dir`: Directory for saved conversations
- `auto_save_conversations`: Auto-save on exit
- `max_history_length`: Maximum conversation turns to keep
- `show_timestamps`: Show timestamps in output
- `show_sources`: Show source citations
- `color_output`: Enable colored terminal output



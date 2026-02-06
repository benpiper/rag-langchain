import os
import logging
import argparse
import sys
import yaml
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_milvus import Milvus
from langchain_community.document_loaders import (
    WebBaseLoader,
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever


def load_config(
    config_path: str = "config.yaml", local_override: str = "config.local.yaml"
) -> Dict[str, Any]:
    """
    Load configuration from YAML file with optional local overrides.

    Args:
        config_path: Path to the main configuration file
        local_override: Path to local override configuration file

    Returns:
        Dictionary containing merged configuration

    Raises:
        FileNotFoundError: If main config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    # Load main config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load local overrides if they exist
    if os.path.exists(local_override):
        with open(local_override, "r") as f:
            local_config = yaml.safe_load(f)
            if local_config:
                # Deep merge - simple version (overwrites nested dicts)
                config = deep_merge(config, local_config)

    return config


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base dictionary
        override: Override dictionary

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# Load configuration
config = load_config()

# Configure logging
logging.basicConfig(
    level=getattr(logging, config["logging"]["level"]),
    format=config["logging"]["format"],
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config["logging"]["log_file"]),
    ],
)
logger = logging.getLogger(__name__)

# Set the chat model
model = ChatOpenAI(
    model=config["llm"]["model"],
    temperature=config["llm"].get("temperature", 0.7),
    max_tokens=config["llm"].get("max_tokens"),
)

# Global variables for embeddings and vector store
embeddings = None
vector_store = None
bm25_retriever = None
ensemble_retriever = None
reranker = None


def setup_reranker():
    """Initialize the cross-encoder reranker."""
    global reranker
    try:
        logger.info("Initializing Cross-Encoder reranker...")
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
        logger.info("Reranker initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize reranker: {e}")


def initialize_hybrid_retriever(all_docs: List[Any]):
    """
    Initialize BM25 and Ensemble retrievers for hybrid search.
    """
    global bm25_retriever, ensemble_retriever

    if not all_docs:
        logger.warning("No documents provided for hybrid retriever initialization")
        return

    logger.info(f"Initializing hybrid retriever with {len(all_docs)} documents...")

    try:
        # Initialize BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = config["retrieval"]["k"]

        # Create ensemble retriever (RRF)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[
                bm25_retriever,
                vector_store.as_retriever(
                    search_kwargs={"k": config["retrieval"]["k"]}
                ),
            ],
            weights=[0.3, 0.7],
        )
        logger.info("Hybrid retriever initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize hybrid retriever: {e}")


def rerank_documents(query: str, docs: List[Any], top_n: int = 5) -> List[Any]:
    """Re-rank retrieved documents using a cross-encoder model."""
    if not reranker or not docs:
        if not reranker:
            logger.warning("Reranker not initialized, skipping re-ranking")
        return docs[:top_n]

    try:
        logger.info(f"Re-ranking {len(docs)} documents...")
        # Include title/source in the text for the reranker to provide better context
        pairs = []
        for doc in docs:
            title = doc.metadata.get("title", "")
            # If title is just a path, take the filename
            if "/" in title or "\\" in title:
                title = os.path.basename(title)

            context_text = f"Title: {title}\nContent: {doc.page_content}"
            pairs.append([query, context_text])

        scores = reranker.predict(pairs)

        for i, doc in enumerate(docs):
            doc.metadata["rerank_score"] = float(scores[i])

        reranked_docs = sorted(
            docs, key=lambda x: x.metadata["rerank_score"], reverse=True
        )
        logger.info(
            f"Re-ranking completed. Top score: {reranked_docs[0].metadata['rerank_score'] if reranked_docs else 'N/A'}"
        )
        return reranked_docs[:top_n]
    except Exception as e:
        logger.error(f"Re-ranking failed: {e}")
        return docs[:top_n]


def generate_multi_queries(query: str, n: int = 3) -> List[str]:
    """Use the LLM to generate multiple search variations of the query."""
    logger.info(f"Generating {n} query variations for: {query}")
    try:
        prompt = (
            f"You are an AI language model assistant. Your task is to generate {n} "
            f"different versions of the given user query to retrieve relevant documents from a vector database. "
            f"By generating multiple perspectives on the user query, your goal is to help "
            f"the user overcome some of the limitations of the distance-based similarity search. "
            f"Provide these alternative searches separated by newlines, with NO other text.\n"
            f"Original query: {query}"
        )

        response = model.invoke(prompt)
        queries = [q.strip() for q in response.content.split("\n") if q.strip()]

        # Ensure original query is present
        if query not in queries:
            queries.insert(0, query)

        logger.info(f"Generated {len(queries)} variations")
        return queries[: n + 1]
    except Exception as e:
        logger.error(f"Failed to generate multi-queries: {e}")
        return [query]


def setup_vector_store(provider="openai", ollama_host=None, ollama_model=None):
    """
    Initialize the vector store with specified embedding provider.

    Args:
        provider: Embedding provider ('openai' or 'ollama')
        ollama_host: Ollama server host (only for ollama provider)
        ollama_model: Ollama model name (only for ollama provider)

    Raises:
        ValueError: If provider is invalid or required parameters are missing
        RuntimeError: If initialization fails
    """
    global embeddings, vector_store

    try:
        if provider == "openai":
            logger.info("Initializing OpenAI embeddings...")
            try:
                embeddings = OpenAIEmbeddings(
                    model=config["embeddings"]["openai"]["model"]
                )
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI embeddings: {e}")
                raise RuntimeError(
                    f"OpenAI initialization failed. Check your API key and network: {e}"
                )

        elif provider == "ollama":
            if not ollama_model:
                ollama_model = config["embeddings"]["ollama"]["model"]

            if not ollama_host:
                ollama_host = config["embeddings"]["ollama"]["host"]

            # Ensure host has protocol
            if not ollama_host.startswith("http"):
                ollama_host = f"http://{ollama_host}"

            # Ensure host has port if not present (heuristic)
            default_port = config["embeddings"]["ollama"]["port"]
            try:
                if ":" not in ollama_host.split("//")[1]:
                    ollama_host = f"{ollama_host}:{default_port}"
            except IndexError:
                logger.error(f"Invalid Ollama host format: {ollama_host}")
                raise ValueError(f"Invalid Ollama host format: {ollama_host}")

            logger.info(
                f"Initializing Ollama embeddings at {ollama_host} with model {ollama_model}..."
            )
            try:
                embeddings = OllamaEmbeddings(base_url=ollama_host, model=ollama_model)
            except Exception as e:
                logger.error(f"Failed to initialize Ollama embeddings: {e}")
                raise RuntimeError(
                    f"Ollama initialization failed. Check if Ollama is running at {ollama_host}: {e}"
                )
        else:
            raise ValueError(
                f"Invalid provider: {provider}. Must be 'openai' or 'ollama'"
            )

        # Initialize the vector store
        logger.info("Initializing Milvus vector store...")
        try:
            vector_store = Milvus(
                embedding_function=embeddings,
                connection_args={"uri": config["vector_store"]["milvus"]["uri"]},
                collection_name=config["vector_store"]["milvus"].get(
                    "collection_name", "LangChainCollection"
                ),
                auto_id=config["vector_store"]["milvus"]["auto_id"],
            )
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Milvus vector store: {e}")
            raise RuntimeError(f"Milvus initialization failed: {e}")

    except Exception as e:
        logger.error(f"Vector store setup failed: {e}")
        raise


# RETRIEVAL WITH RAG AGENT

# Define the tool to fetch docs from the document store


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """
    Retrieve information to help answer a query.

    Args:
        query: The search query string

    Returns:
        Tuple of (serialized context, retrieved documents)

    Raises:
        RuntimeError: If vector store is not initialized or search fails
    """
    if vector_store is None:
        logger.error("Vector store not initialized")
        raise RuntimeError(
            "Vector store not initialized. Call setup_vector_store first."
        )

    if not query or not query.strip():
        logger.warning("Empty query provided to retrieve_context")
        return "No query provided", []

    try:
        logger.debug(f"Searching for: {query}")

        # Generate multiple query variations for broader retrieval
        queries = generate_multi_queries(query)
        all_retrieved_docs = []
        seen_contents = set()

        for q in queries:
            logger.debug(f"Running sub-retrieval for query: {q}")
            # Use ensemble retriever if available, otherwise fallback to vector store
            if ensemble_retriever:
                logger.debug("Using hybrid search (EnsembleRetriever)...")
                sub_docs = ensemble_retriever.invoke(q)
            else:
                logger.debug("Using vector search (similarity_search)...")
                sub_docs = vector_store.similarity_search(q, k=config["retrieval"]["k"])

            # Deduplicate documents based on content
            for doc in sub_docs:
                if doc.page_content not in seen_contents:
                    all_retrieved_docs.append(doc)
                    seen_contents.add(doc.page_content)

        logger.info(
            f"Total unique documents retrieved across {len(queries)} queries: {len(all_retrieved_docs)}"
        )

        # Apply re-ranking on the combined set
        top_docs = rerank_documents(
            query, all_retrieved_docs, top_n=config["retrieval"].get("k", 5)
        )

        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in top_docs
        )
        return serialized, top_docs
    except Exception as e:
        logger.error(f"Failed to retrieve context: {e}")
        raise RuntimeError(f"Context retrieval failed: {e}")


# RETRIEVAL WITH RAG AGENT


def run_agent(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    output_format: str = "plain",
):
    """
    Run the RAG agent with tool-based retrieval.

    Args:
        query: The user's query string
        conversation_history: Optional list of previous messages for context
        output_format: Output format ('plain', 'markdown', 'json')

    Returns:
        Tuple of (response_text, updated_conversation_history)

    Raises:
        ValueError: If query is empty
        RuntimeError: If agent execution fails
    """
    if not query or not query.strip():
        logger.error("Empty query provided to run_agent")
        raise ValueError("Query cannot be empty")

    try:
        # Initialize conversation history if not provided
        if conversation_history is None:
            conversation_history = []

        # Check if conversation history is too long and needs summarization
        max_history = config.get("user_experience", {}).get("max_history_length", 50)
        if len(conversation_history) > max_history:
            logger.info(
                f"Conversation history length ({len(conversation_history)}) exceeds limit ({max_history}). Summarizing..."
            )
            # Keep the last 10 messages for immediate context, summarize the rest
            recent_context = conversation_history[-10:]
            to_summarize = conversation_history[:-10]
            summary = summarize_conversation(to_summarize)

            if summary:
                conversation_history = [
                    {
                        "role": "system",
                        "content": f"The following is a summary of the previous conversation: {summary}",
                    }
                ] + recent_context
                logger.info("History compressed with summary")

        # Create the agent
        tools = [retrieve_context]

        logger.info("Creating RAG agent...")
        agent = create_agent(
            model, tools, system_prompt=config["prompts"]["agent_system_prompt"]
        )

        # Build messages including history
        messages = conversation_history.copy()
        messages.append({"role": "user", "content": query})

        logger.info("Streaming RAG agent response...")
        response_text = ""

        if output_format == "json":
            pass

        for event in agent.stream(
            {"messages": messages},
            stream_mode="values",
        ):
            last_message = event["messages"][-1]

            if output_format == "plain":
                last_message.pretty_print()
            elif output_format == "json":
                pass
            else:  # markdown
                if hasattr(last_message, "content"):
                    content = last_message.content
                    if isinstance(content, str) and content:
                        print(content)

            # Capture response text
            if hasattr(last_message, "content"):
                response_text = last_message.content

        # Update conversation history
        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({"role": "assistant", "content": response_text})

        if output_format == "json":
            output = {
                "query": query,
                "response": response_text,
                "timestamp": datetime.now().isoformat(),
                "conversation_history": conversation_history,
            }
            print(json.dumps(output, indent=2))

        logger.info("Agent execution completed successfully")
        return response_text, conversation_history

    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        raise RuntimeError(f"Failed to run agent: {e}")


# RETRIEVAL WITH RAG CHAINS


@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """
    Inject context into state messages by retrieving relevant documents.

    Args:
        request: The model request containing state and messages

    Returns:
        System message with retrieved context

    Raises:
        RuntimeError: If vector store is not initialized or retrieval fails
    """
    if vector_store is None:
        logger.error("Vector store not initialized in prompt_with_context")
        raise RuntimeError(
            "Vector store not initialized. Call setup_vector_store first."
        )

    try:
        last_query = request.state["messages"][-1].text
        logger.debug(f"Retrieving context for: {last_query}")

        retrieved_docs = vector_store.similarity_search(
            last_query, k=config["retrieval"]["k"]
        )
        logger.info(f"Retrieved {len(retrieved_docs)} documents for context")

        docs_content = "\n\n".join(
            (
                f"Source: {doc.metadata.get('source', 'unknown')}\n"
                f"Title: {doc.metadata.get('title', 'No title')}\n"
                f"Content: {doc.page_content}"
            )
            for doc in retrieved_docs
        )

        system_message = (
            f"{config['prompts']['chain_system_prompt']}\n\n"
            f"Retrieved Documents:\n\n{docs_content}"
        )

        return system_message

    except Exception as e:
        logger.error(f"Failed to retrieve context in prompt_with_context: {e}")
        raise RuntimeError(f"Context injection failed: {e}")


def run_chain(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    output_format: str = "plain",
):
    """
    Run the RAG chain with middleware-based retrieval.

    Args:
        query: The user's query string
        conversation_history: Optional list of previous messages for context
        output_format: Output format ('plain', 'markdown', 'json')

    Returns:
        Tuple of (response_text, updated_conversation_history)

    Raises:
        ValueError: If query is empty
        RuntimeError: If chain execution fails
    """
    if not query or not query.strip():
        logger.error("Empty query provided to run_chain")
        raise ValueError("Query cannot be empty")

    try:
        # Initialize conversation history if not provided
        if conversation_history is None:
            conversation_history = []

        # Check if conversation history is too long and needs summarization
        max_history = config.get("user_experience", {}).get("max_history_length", 50)
        if len(conversation_history) > max_history:
            logger.info(
                f"Conversation history length ({len(conversation_history)}) exceeds limit ({max_history}). Summarizing..."
            )
            # Keep the last 10 messages for immediate context, summarize the rest
            recent_context = conversation_history[-10:]
            to_summarize = conversation_history[:-10]
            summary = summarize_conversation(to_summarize)

            if summary:
                conversation_history = [
                    {
                        "role": "system",
                        "content": f"The following is a summary of the previous conversation: {summary}",
                    }
                ] + recent_context
                logger.info("History compressed with summary")

        logger.info("Creating RAG chain with middleware...")
        agent = create_agent(model, tools=[], middleware=[prompt_with_context])

        # Build messages including history
        messages = conversation_history.copy()
        messages.append({"role": "user", "content": query})

        logger.info("Streaming RAG chain response...")
        response_text = ""

        if output_format == "json":
            pass

        for step in agent.stream(
            {"messages": messages},
            stream_mode="values",
        ):
            last_message = step["messages"][-1]

            if output_format == "plain":
                last_message.pretty_print()
            elif output_format == "json":
                pass
            else:  # markdown
                if hasattr(last_message, "content"):
                    content = last_message.content
                    if isinstance(content, str) and content:
                        print(content)

            # Capture response text
            if hasattr(last_message, "content"):
                response_text = last_message.content

        # Update conversation history
        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({"role": "assistant", "content": response_text})

        if output_format == "json":
            output = {
                "query": query,
                "response": response_text,
                "timestamp": datetime.now().isoformat(),
                "conversation_history": conversation_history,
            }
            print(json.dumps(output, indent=2))

        logger.info("Chain execution completed successfully")
        return response_text, conversation_history

    except Exception as e:
        logger.error(f"Chain execution failed: {e}")
        raise RuntimeError(f"Failed to run chain: {e}")


def save_conversation(
    conversation_history: List[Dict[str, str]], filepath: str = None
) -> str:
    """
    Save conversation history to a JSON file.

    Args:
        conversation_history: List of conversation messages
        filepath: Optional path to save file (auto-generated if not provided)

    Returns:
        Path to the saved file

    Raises:
        IOError: If file cannot be written
    """
    if not filepath:
        # Create conversations directory if it doesn't exist
        conversations_dir = "conversations"
        if not os.path.exists(conversations_dir):
            os.makedirs(conversations_dir)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(conversations_dir, f"conversation_{timestamp}.json")

    try:
        conversation_data = {
            "timestamp": datetime.now().isoformat(),
            "messages": conversation_history,
        }

        with open(filepath, "w") as f:
            json.dump(conversation_data, f, indent=2)

        logger.info(f"Conversation saved to {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")
        raise IOError(f"Failed to save conversation: {e}")


def load_conversation(filepath: str) -> List[Dict[str, str]]:
    """
    Load conversation history from a JSON file.

    Args:
        filepath: Path to the conversation file

    Returns:
        List of conversation messages

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    try:
        with open(filepath, "r") as f:
            conversation_data = json.load(f)

        messages = conversation_data.get("messages", [])
        logger.info(
            f"Loaded conversation with {len(messages)} messages from {filepath}"
        )
        return messages

    except FileNotFoundError:
        logger.error(f"Conversation file not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in conversation file: {e}")
        raise

        raise


def summarize_conversation(conversation_history: List[Dict[str, str]]) -> str:
    """Summarize the conversation history to save tokens."""
    if not conversation_history:
        return ""

    logger.info("Summarizing conversation history...")
    try:
        history_text = "\n".join(
            [f"{m['role']}: {m['content']}" for m in conversation_history]
        )
        prompt = (
            f"Summarize the following conversation history concisely, "
            f"preserving key facts and user intent:\n\n{history_text}"
        )

        response = model.invoke(prompt)
        summary = response.content
        logger.info("Conversation summarized successfully")
        return summary
    except Exception as e:
        logger.error(f"Failed to summarize conversation: {e}")
        return ""


def run_interactive_mode(mode: str = "agent", output_format: str = "plain"):
    """
    Run the application in interactive mode with conversation history.

    Args:
        mode: 'agent' or 'chain'
        output_format: Output format ('plain', 'markdown', 'json')

    Raises:
        RuntimeError: If execution fails
    """
    conversation_history = []

    print("\n" + "=" * 60)
    print("RAG Interactive Mode")
    print("=" * 60)
    print(f"Mode: {mode.upper()}")
    print(f"Output format: {output_format}")
    print("\nCommands:")
    print("  /help     - Show this help message")
    print("  /history  - Show conversation history")
    print("  /save     - Save conversation to file")
    print("  /load     - Load conversation from file")
    print("  /clear    - Clear conversation history")
    print("  /exit     - Exit interactive mode")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("\n> ").strip()

            if not query:
                continue

            # Handle commands
            if query == "/exit":
                print("\nExiting interactive mode...")
                break

            elif query == "/help":
                print("\nCommands:")
                print("  /help     - Show this help message")
                print("  /history  - Show conversation history")
                print("  /sources  - List indexed documents")
                print("  /save     - Save conversation to file")
                print("  /load     - Load conversation from file")
                print("  /clear    - Clear conversation history")
                print("  /exit     - Exit interactive mode")
                continue

            elif query == "/history":
                if not conversation_history:
                    print("\nNo conversation history yet.")
                else:
                    print(
                        f"\nConversation history ({len(conversation_history)} messages):"
                    )
                    for i, msg in enumerate(conversation_history, 1):
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        preview = (
                            content[:100] + "..." if len(content) > 100 else content
                        )
                        print(f"{i}. [{role}]: {preview}")
                continue

            elif query == "/save":
                if not conversation_history:
                    print("\nNo conversation to save.")
                else:
                    try:
                        filepath = save_conversation(conversation_history)
                        print(f"\n✓ Conversation saved to: {filepath}")
                    except Exception as e:
                        print(f"\n✗ Failed to save conversation: {e}")
                continue

            elif query == "/sources":
                print("\nFetching unique sources from index...")
                try:
                    from pymilvus import MilvusClient

                    client = MilvusClient(config["vector_store"]["milvus"]["uri"])

                    # Try to find the correct collection
                    target_collection = config["vector_store"]["milvus"].get(
                        "collection_name", "LangChainCollection"
                    )

                    existing_collections = client.list_collections()

                    if target_collection not in existing_collections:
                        # Fallback to LangChainCollection if it exists
                        if "LangChainCollection" in existing_collections:
                            target_collection = "LangChainCollection"
                        elif existing_collections:
                            # Use the first one found as a last resort
                            target_collection = existing_collections[0]
                        else:
                            print(
                                "✗ No collections found in the database. Have you indexed documents yet?"
                            )
                            continue

                    # Query unique sources
                    res = client.query(
                        collection_name=target_collection,
                        output_fields=["source"],
                        filter="",
                        limit=16384,
                    )
                    sources = sorted(list(set(r.get("source", "Unknown") for r in res)))
                    if not sources:
                        print(
                            f"No documents found in collection '{target_collection}'."
                        )
                    else:
                        print(
                            f"\nIndexed Sources ({len(sources)} unique files) from '{target_collection}':"
                        )
                        for s in sources:
                            print(f"  - {s}")
                except Exception as e:
                    print(f"\n✗ Failed to fetch sources: {e}")
                continue

            elif query == "/load":
                filepath = input("Enter conversation file path: ").strip()
                try:
                    conversation_history = load_conversation(filepath)
                    print(f"\n✓ Loaded {len(conversation_history)} messages")
                except Exception as e:
                    print(f"\n✗ Failed to load conversation: {e}")
                continue

            elif query == "/clear":
                conversation_history = []
                print("\n✓ Conversation history cleared")
                continue

            # Run query
            try:
                if mode == "agent":
                    response_text, conversation_history = run_agent(
                        query, conversation_history, output_format
                    )
                else:
                    response_text, conversation_history = run_chain(
                        query, conversation_history, output_format
                    )
            except Exception as e:
                print(f"\n✗ Error: {e}")
                logger.error(f"Query execution failed: {e}")

        except (EOFError, KeyboardInterrupt):
            print("\n\nExiting interactive mode...")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG application")
    parser.add_argument(
        "--mode",
        choices=["agent", "chain"],
        default=config["defaults"]["mode"],
        help=f"Retrieval mode (default: {config['defaults']['mode']})",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query to run (optional, will start interactive mode if not provided)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive mode with conversation history",
    )
    parser.add_argument(
        "--output-format",
        choices=["plain", "markdown", "json"],
        default="plain",
        help="Output format (default: plain)",
    )
    parser.add_argument(
        "--load-conversation", type=str, help="Load conversation history from file"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        default=config["defaults"]["force_refresh"],
        help="Force re-indexing of documents",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["openai", "ollama"],
        default=config["defaults"]["embedding_provider"],
        help=f"Embedding provider (default: {config['defaults']['embedding_provider']})",
    )
    parser.add_argument(
        "--ollama-host",
        default=config["embeddings"]["ollama"]["host"],
        help=f"Ollama host (default: {config['embeddings']['ollama']['host']})",
    )
    parser.add_argument(
        "--ollama-model",
        default=config["embeddings"]["ollama"]["model"],
        help=f"Ollama model (default: {config['embeddings']['ollama']['model']})",
    )
    args = parser.parse_args()

    # Initialize vector store based on arguments
    try:
        setup_vector_store(
            provider=args.embedding_provider,
            ollama_host=args.ollama_host,
            ollama_model=args.ollama_model,
        )
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        sys.exit(1)

    # Initialize reranker
    setup_reranker()

    # Handle force refresh
    if args.force_refresh:
        logger.info("Force refresh requested. Dropping collection...")
        try:
            # Try multiple methods to drop the collection
            dropped = False

            if hasattr(vector_store, "client"):
                try:
                    # For MilvusClient (pymilvus v2.4+)
                    vector_store.client.drop_collection(vector_store.collection_name)
                    logger.info("Collection dropped via client method")
                    dropped = True
                except Exception as e:
                    logger.debug(f"Client drop method failed: {e}")

            if not dropped and hasattr(vector_store, "col"):
                try:
                    # For older pymilvus Collection object
                    vector_store.col.drop()
                    logger.info("Collection dropped via col method")
                    dropped = True
                except Exception as e:
                    logger.debug(f"Col drop method failed: {e}")

            # Fallback: delete the database file
            if not dropped:
                db_path = config["vector_store"]["milvus"]["uri"]
                if os.path.exists(db_path):
                    try:
                        os.remove(db_path)
                        logger.info(f"Deleted {db_path} file")
                        dropped = True
                    except OSError as e:
                        logger.error(f"Failed to delete {db_path}: {e}")

            if not dropped:
                logger.warning("Could not drop collection using any method")

        except Exception as e:
            logger.error(f"Error during force refresh: {e}")
            logger.warning("Continuing with existing vector store...")

    # Check if we need to ingest data (simple check: is the store empty?)
    # We check if force_refresh is True OR if the search returns nothing.
    should_index = args.force_refresh
    if not should_index:
        try:
            logger.info("Checking if vector store contains data...")
            results = vector_store.similarity_search(
                "test", k=config["retrieval"]["test_k"]
            )
            if not results:
                logger.info("Vector store is empty")
                should_index = True
            else:
                logger.info(
                    f"Vector store contains data ({len(results)} documents found)"
                )
        except Exception as e:
            logger.warning(f"Failed to check vector store contents: {e}")
            logger.info("Assuming indexing is needed")
            should_index = True

    if should_index:
        logger.info("Indexing documents...")

        # Load documents from URLs
        urls = []
        sources_file = config["document_processing"]["sources_file"]
        if os.path.exists(sources_file):
            try:
                with open(sources_file, "r") as f:
                    urls = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(urls)} URLs from {sources_file}")
            except Exception as e:
                logger.error(f"Failed to read {sources_file}: {e}")
        else:
            logger.warning(f"{sources_file} not found. Using default URLs.")
            urls = config["document_processing"]["default_urls"]

        # Load web documents with error handling
        web_docs = []
        if urls:
            try:
                logger.info(f"Loading {len(urls)} web documents...")
                loader = WebBaseLoader(web_paths=tuple(urls))
                web_docs = loader.load()
                logger.info(f"Successfully loaded {len(web_docs)} web documents")
            except Exception as e:
                logger.error(f"Failed to load web documents: {e}")
                logger.warning("Continuing without web documents...")
        else:
            logger.warning("No URLs to load")

        # Load local documents
        local_docs_dir = config["document_processing"]["local_docs_dir"]
        logger.info(f"Loading local documents from {local_docs_dir}...")
        local_docs = []

        try:
            if not os.path.exists(local_docs_dir):
                os.makedirs(local_docs_dir)
                logger.info(f"Created {local_docs_dir} directory")

            # Load files based on supported formats
            for glob_pattern in config["document_processing"]["supported_formats"]:
                try:
                    # Select appropriate loader based on file extension
                    if glob_pattern.endswith(".pdf"):
                        loader_cls = PyPDFLoader
                    elif glob_pattern.endswith(".docx"):
                        loader_cls = Docx2txtLoader
                    elif glob_pattern.endswith(".csv"):
                        loader_cls = CSVLoader
                    else:
                        loader_cls = TextLoader

                    logger.debug(
                        f"Using {loader_cls.__name__} for pattern {glob_pattern}"
                    )
                    loader = DirectoryLoader(
                        local_docs_dir,
                        glob=glob_pattern,
                        loader_cls=loader_cls,
                        recursive=True,
                    )
                    # Get list of files that match for logging
                    import pathlib

                    matched_files = list(
                        pathlib.Path(local_docs_dir).rglob(glob_pattern)
                    )
                    if matched_files:
                        logger.info(
                            f"Found {len(matched_files)} files matching {glob_pattern}:"
                        )
                        for f in matched_files[:10]:
                            logger.info(f"  - {f.name}")
                        if len(matched_files) > 10:
                            logger.info(f"  - ... and {len(matched_files) - 10} more")

                    docs = loader.load()
                    local_docs.extend(docs)
                    logger.info(
                        f"Loaded {len(docs)} total pages/documents from {len(matched_files)} files matching {glob_pattern}"
                    )
                except Exception as e:
                    logger.error(f"Failed to load files matching {glob_pattern}: {e}")

            logger.info(f"Total local documents loaded: {len(local_docs)}")

        except Exception as e:
            logger.error(f"Error during local document loading: {e}")
            logger.warning("Continuing without local documents...")

        # Normalize metadata for all documents to ensure consistent schema
        try:
            logger.info("Normalizing document metadata...")
            # Add missing fields to local docs (description, language)
            for doc in local_docs:
                if "title" not in doc.metadata:
                    doc.metadata["title"] = doc.metadata.get("source", "Local Document")
                if "description" not in doc.metadata:
                    doc.metadata["description"] = ""
                if "language" not in doc.metadata:
                    doc.metadata["language"] = "en"

            # Ensure web docs have all fields (they should, but just in case)
            for doc in web_docs:
                if "title" not in doc.metadata:
                    doc.metadata["title"] = doc.metadata.get("source", "Web Document")
                if "description" not in doc.metadata:
                    doc.metadata["description"] = ""
                if "language" not in doc.metadata:
                    doc.metadata["language"] = "en"

            docs = web_docs + local_docs

            if not docs:
                logger.error("No documents loaded. Cannot proceed with indexing.")
                logger.error(
                    f"Please add documents to {sources_file} or {local_docs_dir} directory"
                )
                sys.exit(1)

            logger.info(f"Total documents to process: {len(docs)}")
            logger.debug(f"Total characters in first doc: {len(docs[0].page_content)}")
            if docs and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"First doc preview: {docs[0].page_content[:500]}")

        except Exception as e:
            logger.error(f"Failed to normalize metadata: {e}")
            sys.exit(1)

        # Split documents
        try:
            logger.info("Splitting documents into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config["document_processing"]["chunk_size"],
                chunk_overlap=config["document_processing"]["chunk_overlap"],
                add_start_index=config["document_processing"]["add_start_index"],
            )
            all_splits = text_splitter.split_documents(docs)
            logger.info(f"Split into {len(all_splits)} sub-documents")

            if not all_splits:
                logger.error("Document splitting produced no chunks")
                sys.exit(1)

        except Exception as e:
            logger.error(f"Failed to split documents: {e}")
            sys.exit(1)

        # Embed and store
        try:
            logger.info("Embedding and storing documents in vector store...")
            document_ids = vector_store.add_documents(documents=all_splits)
            logger.info(
                f"Successfully stored {len(document_ids)} documents in vector store"
            )

            # Initialize hybrid search with the new splits
            initialize_hybrid_retriever(all_splits)

            logger.debug(
                f"Document IDs: {document_ids[:10]}..."
                if len(document_ids) > 10
                else f"Document IDs: {document_ids}"
            )

        except Exception as e:
            logger.error(f"Failed to embed and store documents: {e}")
            logger.error(
                "This could be due to API failures, network issues, or vector store errors"
            )
            sys.exit(1)
    else:
        logger.info("Vector store already contains data. Skipping indexing.")
        try:
            # Initialize hybrid search with existing data if possible
            # We fetch a sample for the BM25 index
            logger.info(
                "Fetching existing documents for hybrid search initialization..."
            )
            existing_docs = vector_store.similarity_search(" ", k=1000)
            if existing_docs:
                initialize_hybrid_retriever(existing_docs)
            else:
                logger.warning(
                    "No documents found in vector store to initialize hybrid search"
                )
        except Exception as e:
            logger.warning(
                f"Could not initialize hybrid search from existing data: {e}"
            )

    # Load conversation history if specified
    conversation_history = []
    if args.load_conversation:
        try:
            conversation_history = load_conversation(args.load_conversation)
            logger.info(
                f"Loaded conversation history with {len(conversation_history)} messages"
            )
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            sys.exit(1)

    # Determine mode of operation
    if args.interactive or (not args.query and not args.load_conversation):
        # Interactive mode
        try:
            run_interactive_mode(mode=args.mode, output_format=args.output_format)
        except KeyboardInterrupt:
            logger.info("\nExecution interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Interactive mode failed: {e}")
            sys.exit(1)

    else:
        # Single query mode
        query = args.query

        if not query:
            logger.error("Query cannot be empty in single query mode")
            sys.exit(1)

        # Run the selected mode
        try:
            if args.mode == "agent":
                response_text, conversation_history = run_agent(
                    query, conversation_history, args.output_format
                )
            elif args.mode == "chain":
                response_text, conversation_history = run_chain(
                    query, conversation_history, args.output_format
                )
            else:
                logger.error(f"Invalid mode: {args.mode}")
                sys.exit(1)

        except KeyboardInterrupt:
            logger.info("\nExecution interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            sys.exit(1)

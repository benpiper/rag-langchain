import os
import logging
import argparse
import sys
from typing import Optional, List, Tuple
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_milvus import Milvus
from langchain_community.document_loaders import (
    WebBaseLoader,
    DirectoryLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rag_app.log')
    ]
)
logger = logging.getLogger(__name__)

# Set the chat model
model = ChatOpenAI(model="gpt-4.1")

# Global variables for embeddings and vector store
embeddings = None
vector_store = None


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
                embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI embeddings: {e}")
                raise RuntimeError(
                    f"OpenAI initialization failed. Check your API key and network: {e}"
                )

        elif provider == "ollama":
            if not ollama_model:
                raise ValueError("ollama_model is required when using ollama provider")

            if not ollama_host:
                ollama_host = "http://192.168.88.86:11434"

            # Ensure host has protocol
            if not ollama_host.startswith("http"):
                ollama_host = f"http://{ollama_host}"

            # Ensure host has port if not present (heuristic)
            try:
                if ":" not in ollama_host.split("//")[1]:
                    ollama_host = f"{ollama_host}:11434"
            except IndexError:
                logger.error(f"Invalid Ollama host format: {ollama_host}")
                raise ValueError(f"Invalid Ollama host format: {ollama_host}")

            logger.info(f"Initializing Ollama embeddings at {ollama_host} with model {ollama_model}...")
            try:
                embeddings = OllamaEmbeddings(base_url=ollama_host, model=ollama_model)
            except Exception as e:
                logger.error(f"Failed to initialize Ollama embeddings: {e}")
                raise RuntimeError(
                    f"Ollama initialization failed. Check if Ollama is running at {ollama_host}: {e}"
                )
        else:
            raise ValueError(f"Invalid provider: {provider}. Must be 'openai' or 'ollama'")

        # Initialize the vector store
        logger.info("Initializing Milvus vector store...")
        try:
            vector_store = Milvus(
                embedding_function=embeddings,
                connection_args={"uri": "./milvus_demo.db"},
                auto_id=True,
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
        raise RuntimeError("Vector store not initialized. Call setup_vector_store first.")

    if not query or not query.strip():
        logger.warning("Empty query provided to retrieve_context")
        return "No query provided", []

    try:
        logger.debug(f"Searching for: {query}")
        retrieved_docs = vector_store.similarity_search(query, k=6)
        logger.info(f"Retrieved {len(retrieved_docs)} documents")

        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    except Exception as e:
        logger.error(f"Failed to retrieve context: {e}")
        raise RuntimeError(f"Context retrieval failed: {e}")


# RETRIEVAL WITH RAG AGENT


def run_agent(query: str):
    """
    Run the RAG agent with tool-based retrieval.

    Args:
        query: The user's query string

    Raises:
        ValueError: If query is empty
        RuntimeError: If agent execution fails
    """
    if not query or not query.strip():
        logger.error("Empty query provided to run_agent")
        raise ValueError("Query cannot be empty")

    try:
        # Create the agent
        tools = [retrieve_context]
        # If desired, specify custom instructions
        PROMPT = (
            "You are a helpful assistant with access to a specialized knowledge base. "
            "You MUST use the retrieve_context tool to search for relevant information before answering queries. "
            "When presenting information from the retrieved documents, you MUST cite the source using the URL or Title from the metadata. "
            "Format citations like this: (Source: URL or Title). "
            "If after searching, the retrieved documents do not contain relevant information, state: "
            "'The retrieved documents do not contain specific information about this topic.' "
            "In that case, you may supplement with general knowledge, but make it clear which information came from the documents vs. general knowledge."
        )

        logger.info("Creating RAG agent...")
        agent = create_agent(model, tools, system_prompt=PROMPT)

        logger.info("Streaming RAG agent response...")
        for event in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            event["messages"][-1].pretty_print()

        logger.info("Agent execution completed successfully")

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
        raise RuntimeError("Vector store not initialized. Call setup_vector_store first.")

    try:
        last_query = request.state["messages"][-1].text
        logger.debug(f"Retrieving context for: {last_query}")

        retrieved_docs = vector_store.similarity_search(last_query, k=6)
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
            "You are a helpful assistant. Use the following retrieved documents to answer the user's query. "
            "IMPORTANT: You MUST cite sources when using information from the documents. "
            "Format citations like this: (Source: URL or Title). "
            "If the retrieved documents do not contain relevant information, state: "
            "'The retrieved documents do not contain specific information about this topic.'\n\n"
            f"Retrieved Documents:\n\n{docs_content}"
        )

        return system_message

    except Exception as e:
        logger.error(f"Failed to retrieve context in prompt_with_context: {e}")
        raise RuntimeError(f"Context injection failed: {e}")


def run_chain(query: str):
    """
    Run the RAG chain with middleware-based retrieval.

    Args:
        query: The user's query string

    Raises:
        ValueError: If query is empty
        RuntimeError: If chain execution fails
    """
    if not query or not query.strip():
        logger.error("Empty query provided to run_chain")
        raise ValueError("Query cannot be empty")

    try:
        logger.info("Creating RAG chain with middleware...")
        agent = create_agent(model, tools=[], middleware=[prompt_with_context])

        logger.info("Streaming RAG chain response...")
        for step in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            step["messages"][-1].pretty_print()

        logger.info("Chain execution completed successfully")

    except Exception as e:
        logger.error(f"Chain execution failed: {e}")
        raise RuntimeError(f"Failed to run chain: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG application")
    parser.add_argument(
        "--mode",
        choices=["agent", "chain"],
        default="agent",
        help="Retrieval mode (default: agent)",
    )
    parser.add_argument(
        "--query", type=str, help="Query to run (optional, will prompt if not provided)"
    )
    parser.add_argument(
        "--force-refresh", action="store_true", help="Force re-indexing of documents"
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["openai", "ollama"],
        default="openai",
        help="Embedding provider (default: openai)",
    )
    parser.add_argument(
        "--ollama-host",
        default="192.168.88.86",
        help="Ollama host (default: 192.168.88.86)",
    )
    parser.add_argument(
        "--ollama-model",
        default="embeddinggemma",
        help="Ollama model (default: embeddinggemma)",
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
                db_path = "./milvus_demo.db"
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
            results = vector_store.similarity_search("test", k=1)
            if not results:
                logger.info("Vector store is empty")
                should_index = True
            else:
                logger.info(f"Vector store contains data ({len(results)} documents found)")
        except Exception as e:
            logger.warning(f"Failed to check vector store contents: {e}")
            logger.info("Assuming indexing is needed")
            should_index = True

    if should_index:
        logger.info("Indexing documents...")

        # Load documents from URLs
        urls = []
        if os.path.exists("sources.txt"):
            try:
                with open("sources.txt", "r") as f:
                    urls = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(urls)} URLs from sources.txt")
            except Exception as e:
                logger.error(f"Failed to read sources.txt: {e}")
        else:
            logger.warning("sources.txt not found. Using default URLs.")
            urls = [
                "https://benpiper.com/articles/biblical-creation-account-genesis-theory-evolution/",
                "https://benpiper.com/articles/what-evolution-isnt/",
            ]

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
        logger.info("Loading local documents from ./docs...")
        local_docs = []

        try:
            if not os.path.exists("./docs"):
                os.makedirs("./docs")
                logger.info("Created ./docs directory")

            # Load .txt files
            try:
                local_loader = DirectoryLoader("./docs", glob="**/*.txt", loader_cls=TextLoader)
                txt_docs = local_loader.load()
                local_docs.extend(txt_docs)
                logger.info(f"Loaded {len(txt_docs)} .txt files")
            except Exception as e:
                logger.error(f"Failed to load .txt files: {e}")

            # Load .md files
            try:
                md_loader = DirectoryLoader("./docs", glob="**/*.md", loader_cls=TextLoader)
                md_docs = md_loader.load()
                local_docs.extend(md_docs)
                logger.info(f"Loaded {len(md_docs)} .md files")
            except Exception as e:
                logger.error(f"Failed to load .md files: {e}")

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
                logger.error("Please add documents to sources.txt or ./docs/ directory")
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
                chunk_size=1000,  # chunk size (characters)
                chunk_overlap=200,  # chunk overlap (characters)
                add_start_index=True,  # track index in original document
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
            logger.info(f"Successfully stored {len(document_ids)} documents in vector store")
            logger.debug(f"Document IDs: {document_ids[:10]}..." if len(document_ids) > 10 else f"Document IDs: {document_ids}")

        except Exception as e:
            logger.error(f"Failed to embed and store documents: {e}")
            logger.error("This could be due to API failures, network issues, or vector store errors")
            sys.exit(1)
    else:
        logger.info("Vector store already contains data. Skipping indexing.")

    # Get query from user
    if args.query:
        query = args.query
    else:
        try:
            print("\nEnter your query: ")
            query = input().strip()
        except (EOFError, KeyboardInterrupt):
            logger.info("\nQuery input cancelled by user")
            sys.exit(0)

    if not query:
        logger.error("Query cannot be empty")
        sys.exit(1)

    # Run the selected mode
    try:
        if args.mode == "agent":
            run_agent(query)
        elif args.mode == "chain":
            run_chain(query)
        else:
            logger.error(f"Invalid mode: {args.mode}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\nExecution interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)

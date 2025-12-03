import os
import logging
import argparse
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

# Set the chat model
model = ChatOpenAI(model="gpt-4.1")

# Global variables for embeddings and vector store
embeddings = None
vector_store = None


def setup_vector_store(provider="openai", ollama_host=None, ollama_model=None):
    global embeddings, vector_store

    if provider == "openai":
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    elif provider == "ollama":
        if not ollama_host:
            ollama_host = "http://192.168.88.86:11434"
        # Ensure host has protocol
        if not ollama_host.startswith("http"):
            ollama_host = f"http://{ollama_host}"
        # Ensure host has port if not present (heuristic)
        if ":" not in ollama_host.split("//")[1]:
            ollama_host = f"{ollama_host}:11434"

        embeddings = OllamaEmbeddings(base_url=ollama_host, model=ollama_model)

    # Initialize the vector store
    # Use a local file for Milvus Lite
    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": "./milvus_demo.db"},
        auto_id=True,
    )


# RETRIEVAL WITH RAG AGENT

# Define the tool to fetch docs from the document store


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=6)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# RETRIEVAL WITH RAG AGENT


def run_agent(query: str):
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
    agent = create_agent(model, tools, system_prompt=PROMPT)
    logging.info("RAG agent response")
    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()


# RETRIEVAL WITH RAG CHAINS


@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query, k=6)

    docs_content = "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'unknown')}\nTitle: {doc.metadata.get('title', 'No title')}\nContent: {doc.page_content}"
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


def run_chain(query: str):
    agent = create_agent(model, tools=[], middleware=[prompt_with_context])
    logging.info("RAG chains response")
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()


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
    setup_vector_store(
        provider=args.embedding_provider,
        ollama_host=args.ollama_host,
        ollama_model=args.ollama_model,
    )

    # Handle force refresh
    if args.force_refresh:
        logging.info("Force refresh requested. Dropping collection...")
        # Access the internal collection and drop it.
        # Note: LangChain's Milvus wrapper doesn't expose a direct 'drop_collection' method easily
        # without accessing the internal 'col' or 'client'.
        # However, we can use the pymilvus client directly or try to assume the collection name.
        # The default collection name in LangChain Milvus is 'LangChainCollection'.

        # A safer way with the initialized vector_store:
        # vector_store.col is the Collection object in older versions, or we can use the alias.
        # Let's try to just delete the file if it's local, OR use the proper method if available.
        # Since we are using Milvus Lite with a local file, deleting the file is actually the most robust "hard reset".
        # BUT the user asked for "cleanly delete the store" which implies drop_collection.

        # Let's try to use the internal client to drop.
        try:
            # This depends on the version of langchain-milvus and pymilvus.
            # Assuming vector_store.client is the MilvusClient or similar.
            if hasattr(vector_store, "client"):
                # For MilvusClient (pymilvus v2.4+)
                vector_store.client.drop_collection(vector_store.collection_name)
            elif hasattr(vector_store, "col"):
                # For older pymilvus Collection object
                vector_store.col.drop()

            logging.info("Collection dropped successfully.")
        except Exception as e:
            logging.warning(
                f"Failed to drop collection via client: {e}. Attempting file deletion fallback."
            )
            if os.path.exists("./milvus_demo.db"):
                os.remove("./milvus_demo.db")
                logging.info("Deleted milvus_demo.db file.")

        # Re-initialize vector store after drop (if needed, though usually the object persists)
        # For local file, if we deleted it, we need to ensure the connection is fresh.
        # If we just dropped the collection, we are good to go.

    # Check if we need to ingest data (simple check: is the store empty?)
    # We check if force_refresh is True OR if the search returns nothing.
    should_index = args.force_refresh
    if not should_index:
        results = vector_store.similarity_search("muscle", k=1)
        if not results:
            should_index = True

    if should_index:
        logging.info("Indexing documents...")
        # Load documents from URLs
        # Load URLs from sources.txt
        if os.path.exists("sources.txt"):
            with open("sources.txt", "r") as f:
                urls = [line.strip() for line in f if line.strip()]
        else:
            logging.warning("sources.txt not found. Using default URLs.")
            urls = [
                "https://benpiper.com/articles/biblical-creation-account-genesis-theory-evolution/",
                "https://benpiper.com/articles/what-evolution-isnt/",
            ]

        loader = WebBaseLoader(web_paths=tuple(urls))
        web_docs = loader.load()

        # Load local documents
        logging.info("Loading local documents from ./docs...")
        if not os.path.exists("./docs"):
            os.makedirs("./docs")
            logging.info("Created ./docs directory")

        # Load .txt and .md files
        local_loader = DirectoryLoader("./docs", glob="**/*.txt", loader_cls=TextLoader)
        local_docs = local_loader.load()

        # You can add another loader for .md if needed, or just use glob="**/*" with a generic loader if preferred.
        # For now, let's also try to load .md files using TextLoader (it works for plain text content)
        md_loader = DirectoryLoader("./docs", glob="**/*.md", loader_cls=TextLoader)
        local_docs.extend(md_loader.load())

        # Normalize metadata for all documents to ensure consistent schema
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

        # assert len(docs) == 6  # Removed assertion as count varies with local files
        logging.debug(
            "Total characters in first doc: %s",
            {len(docs[0].page_content) if docs else 0},
        )
        if docs:
            logging.debug(docs[0].page_content[:500])

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # chunk size (characters)
            chunk_overlap=200,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
        all_splits = text_splitter.split_documents(docs)

        logging.info("Split into %s sub-documents.", len(all_splits))

        # Embed and store
        document_ids = vector_store.add_documents(documents=all_splits)
        logging.info("Stored documents in vector store: %s", document_ids)
    else:
        logging.info("Vector store already contains data. Skipping indexing.")

    if args.query:
        query = args.query
    else:
        print("Enter your query: ")
        query = input()

    if args.mode == "agent":
        run_agent(query)
    elif args.mode == "chain":
        run_chain(query)

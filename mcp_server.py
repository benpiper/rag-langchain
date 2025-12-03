import asyncio
import logging
from mcp.server.fastmcp import FastMCP
import rag

# Initialize FastMCP server
mcp = FastMCP("RAG Server")

# Initialize RAG vector store
# We'll use default settings for now, or we could load from env/args
# For simplicity in this server, let's default to OpenAI as it's the primary,
# but we should probably allow configuration.
# Let's try to use the defaults from rag.py which are now dynamic.
# We need to call setup_vector_store.
rag.setup_vector_store(provider="openai")


@mcp.tool()
async def query_knowledge_base(query: str) -> str:
    """
    Query the RAG knowledge base for information.

    Args:
        query: The question or query to search for.

    Returns:
        A string containing relevant information from the knowledge base,
        including citations.
    """
    # We can reuse the logic from rag.py, but we might want to just return the docs
    # or use the agent/chain logic.
    # The user request was "implement an MCP server", implying exposing the RAG capability.
    # Let's expose a simple search first, or maybe the full answer.
    # Let's expose the search capability directly as it's more flexible for an agent.

    if rag.vector_store is None:
        return "Error: Vector store not initialized."

    results = rag.vector_store.similarity_search(query, k=6)
    if not results:
        return "No relevant documents found."

    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'unknown')}\nTitle: {doc.metadata.get('title', 'No title')}\nContent: {doc.page_content}"
        )
        for doc in results
    )
    return serialized


if __name__ == "__main__":
    mcp.run()

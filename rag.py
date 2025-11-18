import os
import logging
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
#from langchain_milvus import Milvus

logging.basicConfig(level=logging.DEBUG)

# Set API key in .env
# os.environ["OPENAI_API_KEY"] = "sk-..."

# Set the chat model
model = ChatOpenAI(model="gpt-4.1")

# Set the embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize the in-memory vector store
vector_store = InMemoryVectorStore(embeddings)

# Initialize the Milvus vector store
""" 
URI = "./milvus_benpiper_blog.db"

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
)
 """
# INDEXING
# Load documents from a URL
# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_="post_content")
loader = WebBaseLoader(
    web_paths=("https://benpiper.com/articles/ask-atheist-where-the-universe-came-from/",
               "https://benpiper.com/articles/matter-proves-the-non-material/",
               "https://benpiper.com/articles/the-people-who-believe-evolution-dont-know-what-it-is-cant-defend-it/",
               "https://benpiper.com/articles/what-evolution-isnt/",
               "https://benpiper.com/articles/atheists-materialists-guilty-nihilism/",
               "https://benpiper.com/articles/biblical-creation-account-genesis-theory-evolution/",
               "https://benpiper.com/articles/why-god-doesnt-stop-pain-evil-suffering/",
               "https://benpiper.com/articles/should-christians-cuss/",
               "https://benpiper.com/articles/libertarian-deception/"
               ),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

assert len(docs) == 9
logging.debug("Total characters: %s", {len(docs[0].page_content)})
logging.debug(docs[0].page_content[:500])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

# Embed and store

document_ids = vector_store.add_documents(documents=all_splits)
logging.info("Stored documents in vector store: %s", document_ids)
#print(document_ids[:3])

# RETRIEVAL WITH RAG AGENT

# Define the tool to fetch docs from the document store

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# Create the agent
tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
    "You have access to a tool that retrieves context from blog posts. "
    "Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)

# Text with a query
query = (
    "Write a short paragraph in the style of these posts.\n\n"
)

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
    retrieved_docs = vector_store.similarity_search(last_query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        "Use the following context in your response:"
        f"\n\n{docs_content}"
    )

    return system_message


agent = create_agent(model, tools=[], middleware=[prompt_with_context])

query = "Write a short paragraph in the style of these posts"
for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()


## Configuring

Put your OpenAI API key in `.env` like so:

`OPENAI_API_KEY=sk-...`

Modify `rag.py` to change the chat model, embeddings model, and URLs you want to index.

## Running

`uv run --env-file .env -- rag.py`

Enter your query when prompted. e.g. "What's the difference between whey and collagen proteins?"


# Streamlit RAG with AWS Bedrock + Pinecone

A minimal query-only RAG UI built with Streamlit. It uses Pinecone as the vector DB and AWS Bedrock for embeddings and generation.

## Prerequisites
- Python 3.10+
- AWS credentials with access to Bedrock (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`)
- Pinecone API key and an existing index (serverless or standard)

## Environment variables
Set these (via shell or a `.env` file):
- `AWS_REGION`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `BEDROCK_MODEL_ID` (e.g., `anthropic.claude-3-sonnet-20240229-v1:0`)
- `BEDROCK_EMBED_MODEL_ID` (e.g., `amazon.titan-embed-text-v1`)
- `PINECONE_API_KEY`
- `PINECONE_ENV` (e.g., `us-east-1-aws`)
- `PINECONE_INDEX` (must exist)

## Setup
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the app
```bash
streamlit run app.py
```
Open the URL Streamlit prints (default http://localhost:8501).

## How it works
- `bedrock/embeddings.py`: gets embeddings from Bedrock (Titan by default).
- `bedrock/vectorstore.py`: queries Pinecone index.
- `bedrock/bedrock.py`: calls a chat model on Bedrock for generation.
- `bedrock/retriever.py`: embeds the query and fetches top-k matches.
- `app.py`: Streamlit UI to submit questions and display answers.

## Notes
- This UI is query-only; no upload/ingest workflow. Use your own pipeline to populate Pinecone.
- Adjust the model IDs in env vars to match models available in your Bedrock region.
- For production, add caching, auth, and better error handling.

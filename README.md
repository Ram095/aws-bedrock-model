# Chat Playground with AWS Bedrock

A simple, interactive chat UI built with Streamlit powered by AWS Bedrock's language models (supporting Claude, Llama3, DeepSeek, and more).

## Prerequisites
- Python 3.10+
- AWS credentials with access to Bedrock (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`)

## Environment variables
Set these (via shell or a `.env` file):
- `AWS_REGION` (e.g., `ap-south-1`)
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `BEDROCK_MODEL_ID` (e.g., `deepseek.v3.2`, `anthropic.claude-3-sonnet-20240229-v1:0`, `meta.llama3-8b-instruct-v1:0`)

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
- `bedrock/bedrock.py`: handles LLM invocation (text generation) via AWS Bedrock API, with support for both text-completion (Llama) and messages-based (Claude, DeepSeek) formats.
- `bedrock/config.py`: loads configuration from environment variables.
- `app.py`: Streamlit UI for interactive chat with message history.

## Features
- **Multi-model support**: Works with Claude, Llama3, DeepSeek, and other AWS Bedrock-compatible models.
- **Flexible conversation**: Chat directly without requiring external context.
- **Context-aware prompts**: Automatically adjusts system prompts based on model type.
- **Persistent chat history**: Messages are maintained during the session.

## Notes
- This is a pure chat interface with no RAG/vector database integration.
- Adjust the `BEDROCK_MODEL_ID` in env vars to match models available in your Bedrock region.
- For production, add authentication, rate limiting, and better error handling.
- Tested with DeepSeek V3 on AWS Bedrock.

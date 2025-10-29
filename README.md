# Customer-Support-AI-Agent
Build a customer bot with Langgraph + Mem0

based on https://docs.mem0.ai/integrations/langgraph

# Init env from scratch

uv init

uv add langgraph langchain-google-genai mem0ai python-dotenv

# Init from .pyproject.toml
uv sync

cp .env.example .env


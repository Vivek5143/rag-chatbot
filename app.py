"""ASGI entry point for the validated Agentic AI eBook RAG API."""

from src.api import create_app


app = create_app()

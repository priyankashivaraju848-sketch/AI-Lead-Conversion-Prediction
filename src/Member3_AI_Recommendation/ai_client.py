
import os
from openai import OpenAI


def create_ai_client():
    """Create OpenRouter client if API key is available."""
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        return None

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

load_dotenv()

try:
    from codemie_sdk import CodeMieClient  # type: ignore
except ImportError:
    try:
        from codemie_sdk_python import CodeMieClient  # type: ignore
    except ImportError:
        CodeMieClient = None  # type: ignore[assignment]


AUTH_SERVER_URL = "https://auth.codemie.lab.epam.com"
AUTH_CLIENT_ID = "codemie-sdk"
AUTH_REALM_NAME = "codemie-prod"
CODEMIE_API_DOMAIN = "https://codemie.lab.epam.com/code-assistant-api"


app = FastAPI(
    title="CodeMie Workshop Proxy",
    description="Simple local proxy for workshop demos.",
    version="0.1.0",
)


class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt text")


@lru_cache(maxsize=1)
def get_codemie_client() -> Any:
    """Create the SDK client once and reuse it across requests."""
    username = os.getenv("CODEMIE_USERNAME", "").strip()
    password = os.getenv("CODEMIE_PASSWORD", "").strip()

    if not username or not password:
        raise HTTPException(
            status_code=500,
            detail="CodeMie credentials are missing. Fill CODEMIE_USERNAME and CODEMIE_PASSWORD in .env.",
        )

    if CodeMieClient is None:
        raise HTTPException(
            status_code=500,
            detail="CodeMie SDK import failed. Install dependencies from requirements.txt.",
        )

    try:
        return CodeMieClient(
            username=username,
            password=password,
            auth_server_url=AUTH_SERVER_URL,
            auth_client_id=AUTH_CLIENT_ID,
            auth_realm_name=AUTH_REALM_NAME,
            codemie_api_domain=CODEMIE_API_DOMAIN,
        )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to initialize the CodeMie client. Check your configuration and credentials.",
        )


def validate_prompt(payload: PromptRequest) -> str:
    """Keep prompt validation explicit and beginner-friendly."""
    prompt = payload.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Prompt must not be empty.")
    return prompt


def normalize_models_response(raw_models: Any) -> list[Any]:
    """Convert the SDK response into a JSON list for FastAPI."""
    encoded = jsonable_encoder(raw_models)

    if isinstance(encoded, list):
        return encoded
    if isinstance(encoded, dict) and isinstance(encoded.get("items"), list):
        return encoded["items"]

    return [encoded]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def list_models() -> list[Any]:
    try:
        client = get_codemie_client()
        models = client.llms.list()
        return normalize_models_response(models)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Could not fetch models from CodeMie.",
        )


@app.post("/echo")
def echo(payload: PromptRequest) -> dict[str, str]:
    prompt = validate_prompt(payload)
    return {"prompt": prompt}


@app.post("/ask")
def ask(payload: PromptRequest) -> dict[str, str]:
    prompt = validate_prompt(payload)

    # Insert the real CodeMie chat/generation call here when the final SDK method is confirmed.
    # Example shape:
    # client = get_codemie_client()
    # result = client.<chat_or_generation_method>(prompt=prompt)

    return {
        "status": "mocked",
        "message": "The proxy works. Connect the real CodeMie generation/chat method here later.",
        "prompt": prompt,
    }

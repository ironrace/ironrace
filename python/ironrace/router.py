"""Async LLM API router for making model calls."""

import asyncio
import json
import os
from typing import Any, Optional

try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


class LLMRouter:
    """Async LLM API caller supporting Claude, OpenAI, and local endpoints.

    This is deliberately thin — LLM calls are I/O-bound and handled in Python.
    The performance-critical context preparation happens in Rust before this.

    Example:
        router = LLMRouter()
        result = await router.call(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
        )
    """

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
    ):
        self.anthropic_api_key = anthropic_api_key or os.environ.get(
            "ANTHROPIC_API_KEY", ""
        )
        self.openai_api_key = openai_api_key or os.environ.get(
            "OPENAI_API_KEY", ""
        )
        self.base_url = base_url
        self.max_retries = max_retries
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        if not HAS_HTTPX:
            raise ImportError(
                "httpx is required for LLM calls: pip install httpx"
            )
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def call(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1500,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Make an LLM API call with retries.

        Dispatches to the appropriate API based on model name prefix.
        """
        if model.startswith("claude") or model.startswith("anthropic"):
            return await self._call_anthropic(
                model, messages, max_tokens, system, **kwargs
            )
        elif model.startswith("gpt") or model.startswith("o1"):
            return await self._call_openai(
                model, messages, max_tokens, system, **kwargs
            )
        elif self.base_url:
            return await self._call_local(
                model, messages, max_tokens, system, **kwargs
            )
        else:
            raise ValueError(
                f"Unknown model '{model}'. Set base_url for local models."
            )

    async def _call_anthropic(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        client = self._get_client()
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            **kwargs,
        }
        if system:
            body["system"] = system

        return await self._request_with_retry(client, url, headers, body)

    async def _call_openai(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        client = self._get_client()
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "content-type": "application/json",
        }
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": all_messages,
            **kwargs,
        }

        return await self._request_with_retry(client, url, headers, body)

    async def _call_local(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        client = self._get_client()
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"content-type": "application/json"}
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": all_messages,
            **kwargs,
        }

        return await self._request_with_retry(client, url, headers, body)

    async def _request_with_retry(
        self,
        client: Any,
        url: str,
        headers: dict,
        body: dict,
    ) -> dict:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await client.post(
                    url, headers=headers, json=body
                )

                if response.status_code == 429:
                    # Rate limited — exponential backoff
                    wait = 2**attempt
                    await asyncio.sleep(wait)
                    continue

                response.raise_for_status()
                return response.json()

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)

        raise RuntimeError(
            f"LLM API call failed after {self.max_retries} retries: {last_error}"
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

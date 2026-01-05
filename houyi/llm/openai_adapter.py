"""OpenAI LLM adapter."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

from houyi.llm.base import LLMAdapter, LLMMessage, LLMResponse


class OpenAIAdapter(LLMAdapter):
    """OpenAI LLM adapter.
    
    Supports GPT-4, GPT-3.5, and other OpenAI models.
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4",
        base_url: str | None = None,
    ):
        """Initialize OpenAI adapter.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            base_url: Optional base URL for API
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        
        # Lazy import to avoid requiring openai package if not used
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. "
                "Install with: pip install openai>=1.0.0"
            )
    
    async def chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Chat completion with OpenAI.
        
        Args:
            messages: Conversation messages
            tools: Available tools (OpenAI function calling format)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI parameters
            
        Returns:
            LLM response
        """
        # Normalize messages
        normalized_messages = self._normalize_messages(messages)
        
        # Build request parameters
        params = {
            "model": self.model,
            "messages": normalized_messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        if tools:
            params["tools"] = tools
        
        # Add any additional parameters
        params.update(kwargs)
        
        # Make API call
        response = await self.client.chat.completions.create(**params)
        
        return LLMResponse.from_openai(response)
    
    async def stream_chat(
        self,
        messages: list[LLMMessage | dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Streaming chat completion with OpenAI.
        
        Args:
            messages: Conversation messages
            tools: Available tools
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI parameters
            
        Yields:
            Response chunks
        """
        # Normalize messages
        normalized_messages = self._normalize_messages(messages)
        
        # Build request parameters
        params = {
            "model": self.model,
            "messages": normalized_messages,
            "temperature": temperature,
            "stream": True,
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        if tools:
            params["tools"] = tools
        
        # Add any additional parameters
        params.update(kwargs)
        
        # Make streaming API call
        stream = await self.client.chat.completions.create(**params)
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import aiohttp
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import aiolimiter


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.limiter = aiolimiter.AsyncLimiter(
            max_rate=config.get('rate_limit_rps', 2),
            time_period=1
        )

    @abstractmethod
    async def generate_json(self, prompt: str, retry_hint: Optional[str] = None) -> Tuple[str, Dict[str, Any], float]:
        """Generate JSON response. Returns (raw_response, usage, latency)."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=config.get('api_key'))

    async def generate_json(self, prompt: str, retry_hint: Optional[str] = None) -> Tuple[str, Dict[str, Any], float]:
        async with self.limiter:
            start_time = time.time()
            messages = [{"role": "user", "content": prompt}]
            if retry_hint:
                messages.append({"role": "assistant", "content": "Invalid JSON"})
                messages.append({"role": "user", "content": retry_hint})

            response = await self.client.chat.completions.create(
                model=self.config['model'],
                messages=messages,
                max_tokens=self.config.get('max_tokens', 1500),
                temperature=self.config.get('temperature', 0.2),
                top_p=self.config.get('top_p', 0.9),
                response_format={"type": "json_object"} if self.config.get('json_mode', True) else None
            )

            latency = time.time() - start_time
            raw_response = response.choices[0].message.content
            usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            return raw_response, usage, latency


class AnthropicClient(LLMClient):
    """Anthropic API client."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = AsyncAnthropic(api_key=config.get('api_key'))

    async def generate_json(self, prompt: str, retry_hint: Optional[str] = None) -> Tuple[str, Dict[str, Any], float]:
        async with self.limiter:
            start_time = time.time()
            system_prompt = None
            user_prompt = prompt
            if retry_hint:
                user_prompt += "\n\n" + retry_hint

            response = await self.client.messages.create(
                model=self.config['model'],
                max_tokens=self.config.get('max_tokens', 1500),
                temperature=self.config.get('temperature', 0.2),
                top_p=self.config.get('top_p', 0.9),
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )

            latency = time.time() - start_time
            raw_response = response.content[0].text
            usage = {
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens
            }
            return raw_response, usage, latency


class LocalClient(LLMClient):
    """Local OpenAI-compatible API client."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('base_url', 'http://localhost:8000/v1')
        self.session = None

    async def _get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def generate_json(self, prompt: str, retry_hint: Optional[str] = None) -> Tuple[str, Dict[str, Any], float]:
        async with self.limiter:
            start_time = time.time()
            session = await self._get_session()
            full_prompt = prompt
            if retry_hint:
                full_prompt += "\n\n" + retry_hint

            payload = {
                "model": self.config['model'],
                "messages": [{"role": "user", "content": full_prompt}],
                "max_tokens": self.config.get('max_tokens', 1500),
                "temperature": self.config.get('temperature', 0.2),
                "top_p": self.config.get('top_p', 0.9),
                "response_format": {"type": "json_object"} if self.config.get('json_mode', True) else None
            }

            async with session.post(f"{self.base_url}/chat/completions", json=payload) as resp:
                result = await resp.json()
                latency = time.time() - start_time
                raw_response = result['choices'][0]['message']['content']
                usage = result.get('usage', {})
                return raw_response, usage, latency

    async def close(self):
        if self.session:
            await self.session.close()


def create_client(config: Dict[str, Any]) -> LLMClient:
    """Factory function for LLM clients."""
    provider = config.get('provider', 'openai')
    if provider == 'openai':
        return OpenAIClient(config)
    elif provider == 'anthropic':
        return AnthropicClient(config)
    elif provider == 'local':
        return LocalClient(config)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


class ConcurrentClient:
    """Handles concurrent LLM calls with semaphore."""

    def __init__(self, client: LLMClient, concurrency: int = 8):
        self.client = client
        self.semaphore = asyncio.Semaphore(concurrency)

    async def generate_json(self, prompt: str, retry_hint: Optional[str] = None) -> Tuple[str, Dict[str, Any], float]:
        async with self.semaphore:
            return await self.client.generate_json(prompt, retry_hint)
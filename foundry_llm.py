import inspect
import os
from typing import AsyncIterator, Iterator, Sequence, Union

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.aio import AIProjectClient as AsyncAIProjectClient
from anthropic import AnthropicFoundry, AsyncAnthropicFoundry
from google import genai
from google.genai import types

Messages = Union[str, Sequence[dict]]

# Verify this scope against the Foundry docs for your tenant/cloud.
_ENTRA_SCOPE = "https://cognitiveservices.azure.com/.default"


def _detect_provider(deployment: str) -> str:
    name = deployment.lower()
    if "claude" in name:
        return "claude"
    if "gemini" in name:
        return "gemini"
    return "openai"


def _normalize(prompt: Messages) -> list[dict]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return list(prompt)


def _split_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
    rest = [m for m in messages if m["role"] != "system"]
    return system, rest


class FoundryLLM:
    """
    llm.invoke / ainvoke / stream / astream for:
      - OpenAI models (Foundry, via azure-ai-projects)
      - Claude models (Foundry, via AnthropicFoundry)
      - Gemini models (custom endpoint, via google-genai)
    """

    def __init__(
        self,
        deployment: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        provider: str | None = None,       # "openai" | "claude" | "gemini"
        claude_api_key: str | None = None,
        gemini_api_key: str | None = None,
        gemini_base_url: str | None = None,
    ):
        self.deployment = deployment
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider or _detect_provider(deployment)

        self._claude_api_key = claude_api_key or os.getenv("ANTHROPIC_FOUNDRY_API_KEY")
        self._gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self._gemini_base_url = gemini_base_url or os.getenv("GEMINI_BASE_URL")

        self._sync_client = None
        self._async_client = None
        self._async_project = None
        self._async_cred = None

    # ================= client construction =================
    def _get_sync_client(self):
        if self._sync_client is not None:
            return self._sync_client

        if self.provider == "gemini":
            self._sync_client = self._make_gemini()
        elif self.provider == "claude":
            kwargs = {"resource": os.getenv("ANTHROPIC_FOUNDRY_RESOURCE")}
            if self._claude_api_key:
                kwargs["api_key"] = self._claude_api_key
            else:
                kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                    DefaultAzureCredential(), _ENTRA_SCOPE
                )
            self._sync_client = AnthropicFoundry(**kwargs)
        else:
            project = AIProjectClient(
                endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
                credential=DefaultAzureCredential(),
            )
            self._sync_client = project.get_openai_client()
        return self._sync_client

    async def _get_async_client(self):
        if self._async_client is not None:
            return self._async_client

        if self.provider == "gemini":
            # genai.Client exposes async via `.aio`; one client serves both
            self._async_client = self._make_gemini()
        elif self.provider == "claude":
            kwargs = {"resource": os.getenv("ANTHROPIC_FOUNDRY_RESOURCE")}
            if self._claude_api_key:
                kwargs["api_key"] = self._claude_api_key
            else:
                self._async_cred = AsyncDefaultAzureCredential()

                async def token_provider() -> str:
                    tok = await self._async_cred.get_token(_ENTRA_SCOPE)
                    return tok.token

                kwargs["azure_ad_token_provider"] = token_provider
            self._async_client = AsyncAnthropicFoundry(**kwargs)
        else:
            self._async_cred = AsyncDefaultAzureCredential()
            self._async_project = AsyncAIProjectClient(
                endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
                credential=self._async_cred,
            )
            client = self._async_project.get_openai_client()
            self._async_client = await client if inspect.isawaitable(client) else client
        return self._async_client

    def _make_gemini(self):
        http_options = types.HttpOptions(base_url=self._gemini_base_url) if self._gemini_base_url else None
        return genai.Client(api_key=self._gemini_api_key, http_options=http_options)

    # ================= request building =================
    def _openai_kwargs(self, prompt, **kw) -> dict:
        return dict(
            model=self.deployment,
            messages=_normalize(prompt),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kw,
        )

    def _claude_kwargs(self, prompt, **kw) -> dict:
        system, msgs = _split_system(_normalize(prompt))
        args = dict(
            model=self.deployment,
            messages=msgs,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kw,
        )
        if system:
            args["system"] = system
        return args

    def _gemini_kwargs(self, prompt, **kw) -> dict:
        system, msgs = _split_system(_normalize(prompt))
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in msgs
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            **kw,
        )
        return dict(model=self.deployment, contents=contents, config=config)

    @staticmethod
    def _claude_text(resp) -> str:
        return "".join(b.text for b in resp.content if b.type == "text")

    # ================= sync =================
    def invoke(self, prompt: Messages, **kw) -> str:
        client = self._get_sync_client()
        if self.provider == "gemini":
            return client.models.generate_content(**self._gemini_kwargs(prompt, **kw)).text or ""
        if self.provider == "claude":
            return self._claude_text(client.messages.create(**self._claude_kwargs(prompt, **kw)))
        resp = client.chat.completions.create(**self._openai_kwargs(prompt, **kw))
        return resp.choices[0].message.content

    def stream(self, prompt: Messages, **kw) -> Iterator[str]:
        client = self._get_sync_client()
        if self.provider == "gemini":
            for chunk in client.models.generate_content_stream(**self._gemini_kwargs(prompt, **kw)):
                if chunk.text:
                    yield chunk.text
        elif self.provider == "claude":
            with client.messages.stream(**self._claude_kwargs(prompt, **kw)) as s:
                yield from s.text_stream
        else:
            events = client.chat.completions.create(**self._openai_kwargs(prompt, stream=True, **kw))
            for chunk in events:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

    # ================= async =================
    async def ainvoke(self, prompt: Messages, **kw) -> str:
        client = await self._get_async_client()
        if self.provider == "gemini":
            resp = await client.aio.models.generate_content(**self._gemini_kwargs(prompt, **kw))
            return resp.text or ""
        if self.provider == "claude":
            return self._claude_text(await client.messages.create(**self._claude_kwargs(prompt, **kw)))
        resp = await client.chat.completions.create(**self._openai_kwargs(prompt, **kw))
        return resp.choices[0].message.content

    async def astream(self, prompt: Messages, **kw) -> AsyncIterator[str]:
        client = await self._get_async_client()
        if self.provider == "gemini":
            stream = await client.aio.models.generate_content_stream(**self._gemini_kwargs(prompt, **kw))
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        elif self.provider == "claude":
            async with client.messages.stream(**self._claude_kwargs(prompt, **kw)) as s:
                async for text in s.text_stream:
                    yield text
        else:
            events = await client.chat.completions.create(**self._openai_kwargs(prompt, stream=True, **kw))
            async for chunk in events:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

    # ================= cleanup =================
    async def aclose(self):
        c = self._async_client
        if c is not None:
            if self.provider == "gemini":
                if hasattr(c.aio, "aclose"):
                    await c.aio.aclose()
            elif hasattr(c, "close"):
                await c.close()
        if self._async_project is not None:
            await self._async_project.close()
        if self._async_cred is not None:
            await self._async_cred.close()
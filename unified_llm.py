"""
unified_llm.py

Unified native-SDK LLM abstraction.

Providers:
    - Microsoft Foundry OpenAI
    - Microsoft Foundry Claude
    - Google Gemini

Operations:
    - invoke()
    - stream()
    - ainvoke()
    - astream()

No LangChain dependency.

Usage:

    llm = FoundryLLM("my-gpt-deployment")

    response = llm.invoke(
        "Explain RAG."
    )

    for chunk in llm.stream(
        "Explain agentic RAG."
    ):
        print(chunk.content, end="")


Async:

    llm = FoundryLLM("my-claude-deployment")

    response = await llm.ainvoke(
        "Explain RAG."
    )

    async for chunk in llm.astream(
        "Explain agentic RAG."
    ):
        print(chunk.content, end="")
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    Sequence,
)


# AZURE

from azure.identity import (
    DefaultAzureCredential,
    get_bearer_token_provider,
)

from azure.identity.aio import (
    DefaultAzureCredential as AsyncDefaultAzureCredential,
)

from azure.ai.projects import AIProjectClient
from azure.ai.projects.aio import (
    AIProjectClient as AsyncAIProjectClient,
)


# ANTHROPIC

from anthropic import (
    AnthropicFoundry,
    AsyncAnthropicFoundry,
)


# GOOGLE GEMINI

from google import genai
from google.genai import types


# TYPES

MessageInput = (
    str
    | Sequence["Message"]
    | Sequence[dict[str, Any]]
)


# MESSAGE

@dataclass(frozen=True)
class Message:
    """
    Provider-neutral chat message.

    Example:

        Message(
            role="system",
            content="You are a helpful assistant."
        )

        Message(
            role="user",
            content="Explain RAG."
        )
    """

    role: str
    content: str


# NORMALIZED RESPONSE

@dataclass
class LLMResponse:
    """
    Normalized non-streaming response.

    Provider-specific raw response is retained so that
    advanced provider functionality is not lost.
    """

    content: str

    model: str

    provider: str

    input_tokens: int | None = None

    output_tokens: int | None = None

    total_tokens: int | None = None

    finish_reason: str | None = None

    raw_response: Any = None


# NORMALIZED STREAM CHUNK

@dataclass
class LLMChunk:
    """
    Normalized streaming chunk.

    Most chunks contain only `content`.

    The final chunk can optionally contain usage/
    finish information.
    """

    content: str = ""

    model: str | None = None

    provider: str | None = None

    input_tokens: int | None = None

    output_tokens: int | None = None

    total_tokens: int | None = None

    finish_reason: str | None = None

    is_final: bool = False

    raw_chunk: Any = None


# BASE LLM

class BaseLLM(ABC):
    """
    Provider-neutral LLM contract.

    Every provider implements:

        invoke()
        stream()
        ainvoke()
        astream()
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # --------------------------------------------------------
    # SYNC INVOKE
    # --------------------------------------------------------

    @abstractmethod
    def invoke(
        self,
        messages: MessageInput,
        **kwargs,
    ) -> LLMResponse:
        raise NotImplementedError

    # --------------------------------------------------------
    # SYNC STREAM
    # --------------------------------------------------------

    @abstractmethod
    def stream(
        self,
        messages: MessageInput,
        **kwargs,
    ) -> Iterator[LLMChunk]:
        raise NotImplementedError

    # --------------------------------------------------------
    # ASYNC INVOKE
    # --------------------------------------------------------

    @abstractmethod
    async def ainvoke(
        self,
        messages: MessageInput,
        **kwargs,
    ) -> LLMResponse:
        raise NotImplementedError

    # --------------------------------------------------------
    # ASYNC STREAM
    # --------------------------------------------------------

    @abstractmethod
    async def astream(
        self,
        messages: MessageInput,
        **kwargs,
    ) -> AsyncIterator[LLMChunk]:
        raise NotImplementedError

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    async def aclose(self):
        """
        Provider-specific async cleanup.

        Override when the SDK owns closable resources.
        """
        return None

    # --------------------------------------------------------
    # ASYNC CONTEXT MANAGER
    # --------------------------------------------------------

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type,
        exc,
        traceback,
    ):
        await self.aclose()


# MESSAGE HELPERS

def normalize_messages(
    messages: MessageInput,
) -> list[Message]:

    # --------------------------------------------------------
    # Simple string prompt
    # --------------------------------------------------------

    if isinstance(messages, str):

        return [
            Message(
                role="user",
                content=messages,
            )
        ]

    normalized: list[Message] = []

    for message in messages:

        # Already a Message

        if isinstance(message, Message):

            normalized.append(message)
            continue

        # Dictionary

        if isinstance(message, dict):

            if "role" not in message:
                raise ValueError(
                    "Message is missing 'role'."
                )

            if "content" not in message:
                raise ValueError(
                    "Message is missing 'content'."
                )

            normalized.append(
                Message(
                    role=message["role"],
                    content=message["content"],
                )
            )

            continue

        raise TypeError(
            "Messages must contain Message "
            "objects or dictionaries."
        )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    valid_roles = {
        "system",
        "user",
        "assistant",
    }

    for message in normalized:

        if message.role not in valid_roles:

            raise ValueError(
                f"Unsupported message role: "
                f"{message.role}"
            )

        if not isinstance(
            message.content,
            str,
        ):

            raise TypeError(
                "Currently only string message "
                "content is supported."
            )

    return normalized


def split_system_messages(
    messages: list[Message],
) -> tuple[
    str | None,
    list[Message],
]:

    system_messages = [
        message.content
        for message in messages
        if message.role == "system"
    ]

    remaining = [
        message
        for message in messages
        if message.role != "system"
    ]

    system = (
        "\n\n".join(system_messages)
        if system_messages
        else None
    )

    return system, remaining


# OPENAI / MICROSOFT FOUNDRY

class OpenAILLM(BaseLLM):
    """
    Microsoft Foundry OpenAI implementation.

    Uses:

        AIProjectClient
            -> OpenAI / AsyncOpenAI

    The Foundry project endpoint is configured using:

        AZURE_AI_PROJECT_ENDPOINT
    """

    ENTRA_SCOPE = (
        "https://ai.azure.com/.default"
    )

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.project_endpoint = os.environ[
            "AZURE_AI_PROJECT_ENDPOINT"
        ]

        self._sync_project = None

        self._sync_client = None

        self._async_project = None

        self._async_client = None

        self._async_credential = None

    # SYNC CLIENT

    def _get_sync_client(self):

        if self._sync_client is not None:
            return self._sync_client

        self._sync_project = AIProjectClient(
            endpoint=self.project_endpoint,
            credential=DefaultAzureCredential(),
        )

        self._sync_client = (
            self._sync_project
            .get_openai_client()
        )

        return self._sync_client

    # ASYNC CLIENT

    async def _get_async_client(self):

        if self._async_client is not None:
            return self._async_client

        self._async_credential = (
            AsyncDefaultAzureCredential()
        )

        self._async_project = (
            AsyncAIProjectClient(
                endpoint=self.project_endpoint,
                credential=self._async_credential,
            )
        )

        self._async_client = (
            self._async_project
            .get_openai_client()
        )

        return self._async_client

    # REQUEST

    def _request(
        self,
        messages: MessageInput,
        **kwargs,
    ) -> dict[str, Any]:

        normalized = normalize_messages(
            messages
        )

        request = {
            "model": self.model,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in normalized
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        request.update(kwargs)

        return request

    # RESPONSE

    def _response(
        self,
        response,
    ) -> LLMResponse:

        usage = getattr(
            response,
            "usage",
            None,
        )

        return LLMResponse(
            content=(
                response
                .choices[0]
                .message
                .content
                or ""
            ),
            model=self.model,
            provider="openai",
            input_tokens=(
                getattr(
                    usage,
                    "prompt_tokens",
                    None,
                )
                if usage
                else None
            ),
            output_tokens=(
                getattr(
                    usage,
                    "completion_tokens",
                    None,
                )
                if usage
                else None
            ),
            total_tokens=(
                getattr(
                    usage,
                    "total_tokens",
                    None,
                )
                if usage
                else None
            ),
            finish_reason=(
                getattr(
                    response.choices[0],
                    "finish_reason",
                    None,
                )
            ),
            raw_response=response,
        )

    # INVOKE

    def invoke(
        self,
        messages: MessageInput,
        **kwargs,
    ) -> LLMResponse:

        client = self._get_sync_client()

        response = (
            client.chat.completions.create(
                **self._request(
                    messages,
                    **kwargs,
                )
            )
        )

        return self._response(response)

    # STREAM

    def stream(
        self,
        messages: MessageInput,
        **kwargs,
    ) -> Iterator[LLMChunk]:

        client = self._get_sync_client()

        request = self._request(
            messages,
            **kwargs,
        )

        request["stream"] = True

        events = (
            client.chat.completions.create(
                **request
            )
        )

        for event in events:

            if not event.choices:
                continue

            delta = event.choices[0].delta

            content = (
                getattr(
                    delta,
                    "content",
                    None,
                )
                or ""
            )

            finish_reason = getattr(
                event.choices[0],
                "finish_reason",
                None,
            )

            usage = getattr(
                event,
                "usage",
                None,
            )

            if (
                content
                or finish_reason
                or usage
            ):

                yield LLMChunk(
                    content=content,
                    model=self.model,
                    provider="openai",
                    input_tokens=(
                        getattr(
                            usage,
                            "prompt_tokens",
                            None,
                        )
                        if usage
                        else None
                    ),
                    output_tokens=(
                        getattr(
                            usage,
                            "completion_tokens",
                            None,
                        )
                        if usage
                        else None
                    ),
                    total_tokens=(
                        getattr(
                            usage,
                            "total_tokens",
                            None,
                        )
                        if usage
                        else None
                    ),
                    finish_reason=finish_reason,
                    is_final=(
                        finish_reason is not None
                    ),
                    raw_chunk=event,
                )

    # ASYNC INVOKE

    async def ainvoke(
        self,
        messages: MessageInput,
        **kwargs,
    ) -> LLMResponse:

        client = (
            await self._get_async_client()
        )

        response = await (
            client.chat.completions.create(
                **self._request(
                    messages,
                    **kwargs,
                )
            )
        )

        return self._response(response)

    # ASYNC STREAM

    async def astream(
        self,
        messages: MessageInput,
        **kwargs,
    ) -> AsyncIterator[LLMChunk]:

        client = (
            await self._get_async_client()
        )

        request = self._request(
            messages,
            **kwargs,
        )

        request["stream"] = True

        events = await (
            client.chat.completions.create(
                **request
            )
        )

        async for event in events:

            if not event.choices:
                continue

            delta = event.choices[0].delta

            content = (
                getattr(
                    delta,
                    "content",
                    None,
                )
                or ""
            )

            finish_reason = getattr(
                event.choices[0],
                "finish_reason",
                None,
            )

            usage = getattr(
                event,
                "usage",
                None,
            )

            if (
                content
                or finish_reason
                or usage
            ):

                yield LLMChunk(
                    content=content,
                    model=self.model,
                    provider="openai",
                    input_tokens=(
                        getattr(
                            usage,
                            "prompt_tokens",
                            None,
                        )
                        if usage
                        else None
                    ),
                    output_tokens=(
                        getattr(
                            usage,
                            "completion_tokens",
                            None,
                        )
                        if usage
                        else None
                    ),
                    total_tokens=(
                        getattr(
                            usage,
                            "total_tokens",
                            None,
                        )
                        if usage
                        else None
                    ),
                    finish_reason=finish_reason,
                    is_final=(
                        finish_reason is not None
                    ),
                    raw_chunk=event,
                )

    # CLOSE

    async def aclose(self):

        if self._async_client is not None:

            if hasattr(
                self._async_client,
                "close",
            ):
                await self._async_client.close()

        if self._async_project is not None:

            await self._async_project.close()

        if (
            self._async_credential
            is not None
        ):

            await self._async_credential.close()


# CLAUDE / MICROSOFT FOUNDRY

class ClaudeLLM(BaseLLM):
    """
    Microsoft Foundry Claude implementation.

    Uses:

        AnthropicFoundry
        AsyncAnthropicFoundry

    Configuration:

        ANTHROPIC_FOUNDRY_RESOURCE

    Optional API key:

        ANTHROPIC_FOUNDRY_API_KEY

    If API key is not supplied, Entra ID is used.
    """

    ENTRA_SCOPE = (
        "https://ai.azure.com/.default"
    )

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.resource = os.environ.get(
            "ANTHROPIC_FOUNDRY_RESOURCE"
        )

        self.api_key = os.environ.get(
            "ANTHROPIC_FOUNDRY_API_KEY"
        )

        self._sync_client = None

        self._async_client = None

        self._async_credential = None

    # SYNC CLIENT

    def _get_sync_client(self):

        if self._sync_client is not None:
            return self._sync_client

        kwargs: dict[str, Any] = {}

        if self.resource:
            kwargs["resource"] = self.resource

        if self.api_key:

            kwargs["api_key"] = self.api_key

        else:

            kwargs[
                "azure_ad_token_provider"
            ] = get_bearer_token_provider(
                DefaultAzureCredential(),
                self.ENTRA_SCOPE,
            )

        self._sync_client = (
            AnthropicFoundry(**kwargs)
        )

        return self._sync_client

    # ASYNC CLIENT

    async def _get_async_client(self):

        if self._async_client is not None:
            return self._async_client

        kwargs: dict[str, Any] = {}

        if self.resource:
            kwargs["resource"] = self.resource

        if self.api_key:

            kwargs["api_key"] = self.api_key

        else:

            self._async_credential = (
                AsyncDefaultAzureCredential()
            )

            async def token_provider():

                token = (
                    await self._async_credential
                    .get_token(
                        self.ENTRA_SCOPE
                    )
                )

                return token.token

            kwargs[
                "azure_ad_token_provider"
            ] = token_provider

        self._async_client = (
            AsyncAnthropicFoundry(**kwargs)
        )

        return self._async_client

    # REQUEST

    def _request(
        self,
        messages: MessageInput,
        **kwargs,
    ) -> dict[str, Any]:

        normalized = normalize_messages(
            messages
        )

        system, remaining = (
            split_system_messages(
                normalized
            )
        )

        request = {
            "model": self.model,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in remaining
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if system:
            request["system"] = system

        request.update(kwargs)

        return request

    # RESPONSE TEXT

    @staticmethod
    def _text(response) -> str:

        return "".join(
            block.text
            for block in response.content
            if getattr(
                block,
                "type",
                None,
            ) == "text"
        )

    # RESPONSE

    def _response(
        self,
        response,
    ) -> LLMResponse:

        usage = getattr(
            response,
            "usage",
            None,
        )

        input_tokens = getattr(
            usage,
            "input_tokens",
            None,
        )

        output_tokens = getattr(
            usage,
            "output_tokens",
            None,
        )

        total_tokens = None

        if (
            input_tokens is not None
            and output_tokens is not None
        ):
            total_tokens = (
                input_tokens
                + output_tokens
            )

        return LLMResponse(
            content=self._text(response),
            model=self.model,
            provider="claude",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            finish_reason=getattr(
                response,
                "stop_reason",
                None,
            ),
            raw_response=response,
        )

    # INVOKE

    def invoke(
        self,
        messages: MessageInput,
        **kwargs,
    ) -> LLMResponse:

        client = self._get_sync_client()

        response = (
            client.messages.create(
                **self._request(
                    messages,
                    **kwargs,
                )
            )
        )

        return self._response(response)

    # STREAM

    def stream(
        self,
        messages: MessageInput,
        **kwargs,
    ) -> Iterator[LLMChunk]:

        client = self._get_sync_client()

        with client.messages.stream(
            **self._request(
                messages,
                **kwargs,
            )
        ) as stream:

            for text in stream.text_stream:

                yield LLMChunk(
                    content=text,
                    model=self.model,
                    provider="claude",
                    raw_chunk=text,
                )

    # ASYNC INVOKE

    async def ainvoke(
        self,
        messages: MessageInput,
        **kwargs,
    ) -> LLMResponse:

        client = (
            await self._get_async_client()
        )

        response = await (
            client.messages.create(
                **self._request(
                    messages,
                    **kwargs,
                )
            )
        )

        return self._response(response)

    # ASYNC STREAM

    async def astream(
        self,
        messages: MessageInput,
        **kwargs,
    ) -> AsyncIterator[LLMChunk]:

        client = (
            await self._get_async_client()
        )

        async with (
            client.messages.stream(
                **self._request(
                    messages,
                    **kwargs,
                )
            ) as stream
        ):

            async for text in (
                stream.text_stream
            ):

                yield LLMChunk(
                    content=text,
                    model=self.model,
                    provider="claude",
                    raw_chunk=text,
                )

    # CLOSE

    async def aclose(self):

        if self._async_client is not None:

            if hasattr(
                self._async_client,
                "close",
            ):

                await self._async_client.close()

        if (
            self._async_credential
            is not None
        ):

            await self._async_credential.close()


# GEMINI

class GeminiLLM(BaseLLM):
    """
    Google Gemini implementation.

    Uses google-genai.

    Optional organization endpoint:

        GEMINI_BASE_URL

    Optional API key:

        GEMINI_API_KEY

    Example:

        GEMINI_BASE_URL=https://your-org-endpoint/...
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.api_key = os.environ.get(
            "GEMINI_API_KEY"
        )

        self.base_url = os.environ.get(
            "GEMINI_BASE_URL"
        )

        self._client = None

    # CLIENT

    def _get_client(self):

        if self._client is not None:
            return self._client

        http_options = None

        if self.base_url:

            http_options = types.HttpOptions(
                base_url=self.base_url
            )

        self._client = genai.Client(
            api_key=self.api_key,
            http_options=http_options,
        )

        return self._client

    # REQUEST

    def _request(
        self,
        messages: MessageInput,
        **kwargs,
    ) -> dict[str, Any]:

        normalized = normalize_messages(
            messages
        )

        system, remaining = (
            split_system_messages(
                normalized
            )
        )

        contents = []

        for message in remaining:

            role = (
                "model"
                if message.role == "assistant"
                else "user"
            )

            contents.append(
                types.Content(
                    role=role,
                    parts=[
                        types.Part.from_text(
                            text=message.content
                        )
                    ],
                )
            )

        config_kwargs = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }

        if system:

            config_kwargs[
                "system_instruction"
            ] = system

        config_kwargs.update(kwargs)

        config = (
            types.GenerateContentConfig(
                **config_kwargs
            )
        )

        return {
            "model": self.model,
            "contents": contents,
            "config": config,
        }

    # RESPONSE

    def _response(
        self,
        response,
    ) -> LLMResponse:

        usage = getattr(
            response,
            "usage_metadata",
            None,
        )

        input_tokens = getattr(
            usage,
            "prompt_token_count",
            None,
        )

        output_tokens = getattr(
            usage,
            "candidates_token_count",
            None,
        )

        total_tokens = getattr(
            usage,
            "total_token_count",
            None,
        )

        return LLMResponse(
            content=response.text or "",
            model=self.model,
            provider="gemini",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            raw_response=response,
        )

    # INVOKE

    def invoke(
        self,
        messages: MessageInput,
        **kwargs,
    ) -> LLMResponse:

        client = self._get_client()

        response = (
            client.models.generate_content(
                **self._request(
                    messages,
                    **kwargs,
                )
            )
        )

        return self._response(response)

    # STREAM

    def stream(
        self,
        messages: MessageInput,
        **kwargs,
    ) -> Iterator[LLMChunk]:

        client = self._get_client()

        stream = (
            client.models.generate_content_stream(
                **self._request(
                    messages,
                    **kwargs,
                )
            )
        )

        for chunk in stream:

            if not chunk.text:
                continue

            yield LLMChunk(
                content=chunk.text,
                model=self.model,
                provider="gemini",
                raw_chunk=chunk,
            )

    # ASYNC INVOKE

    async def ainvoke(
        self,
        messages: MessageInput,
        **kwargs,
    ) -> LLMResponse:

        client = self._get_client()

        response = await (
            client.aio.models.generate_content(
                **self._request(
                    messages,
                    **kwargs,
                )
            )
        )

        return self._response(response)

    # ASYNC STREAM

    async def astream(
        self,
        messages: MessageInput,
        **kwargs,
    ) -> AsyncIterator[LLMChunk]:

        client = self._get_client()

        stream = await (
            client.aio.models
            .generate_content_stream(
                **self._request(
                    messages,
                    **kwargs,
                )
            )
        )

        async for chunk in stream:

            if not chunk.text:
                continue

            yield LLMChunk(
                content=chunk.text,
                model=self.model,
                provider="gemini",
                raw_chunk=chunk,
            )

    # CLOSE

    async def aclose(self):

        if self._client is None:
            return

        aio_client = self._client.aio

        if hasattr(
            aio_client,
            "aclose",
        ):

            await aio_client.aclose()


# MODEL PROVIDER DETECTION

def detect_provider(model: str) -> str:
    """
    Determine provider from the deployment/model name.

    Your frontend is already responsible for sending
    the correct deployment name.

    Adjust these rules to match YOUR deployment naming.

    Examples:

        gpt-xxx              -> openai
        o4-xxx               -> openai
        claude-xxx           -> claude
        gemini-xxx           -> gemini
    """

    name = model.lower()

    # --------------------------------------------------------
    # Claude
    # --------------------------------------------------------

    if (
        "claude" in name
        or "anthropic" in name
    ):
        return "claude"

    # --------------------------------------------------------
    # Gemini
    # --------------------------------------------------------

    if "gemini" in name:
        return "gemini"

    # --------------------------------------------------------
    # Everything else defaults to OpenAI
    # --------------------------------------------------------

    return "openai"


# UNIFIED FACADE

class FoundryLLM:
    """
    Main application-facing LLM.

    Your application only needs:

        llm = FoundryLLM(model)

    Then:

        llm.invoke(...)
        llm.stream(...)

        await llm.ainvoke(...)
        async for chunk in llm.astream(...)

    Provider-specific implementation is completely
    hidden behind this facade.
    """

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):

        self.model = model

        provider = detect_provider(model)

        # Provider-specific implementation

        if provider == "openai":

            self._llm = OpenAILLM(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        elif provider == "claude":

            self._llm = ClaudeLLM(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        elif provider == "gemini":

            self._llm = GeminiLLM(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        else:

            raise ValueError(
                f"Unsupported provider: {provider}"
            )

    # PROVIDER

    @property
    def provider(self) -> str:

        return detect_provider(
            self.model
        )

    # INVOKE

    def invoke(
        self,
        messages: MessageInput,
        **kwargs,
    ) -> LLMResponse:

        return self._llm.invoke(
            messages,
            **kwargs,
        )

    # STREAM

    def stream(
        self,
        messages: MessageInput,
        **kwargs,
    ) -> Iterator[LLMChunk]:

        yield from self._llm.stream(
            messages,
            **kwargs,
        )

    # ASYNC INVOKE

    async def ainvoke(
        self,
        messages: MessageInput,
        **kwargs,
    ) -> LLMResponse:

        return await self._llm.ainvoke(
            messages,
            **kwargs,
        )

    # ASYNC STREAM

    async def astream(
        self,
        messages: MessageInput,
        **kwargs,
    ) -> AsyncIterator[LLMChunk]:

        async for chunk in (
            self._llm.astream(
                messages,
                **kwargs,
            )
        ):

            yield chunk

    # CLOSE

    async def aclose(self):

        await self._llm.aclose()

    # ASYNC CONTEXT MANAGER

    async def __aenter__(self):

        return self

    async def __aexit__(
        self,
        exc_type,
        exc,
        traceback,
    ):

        await self.aclose()
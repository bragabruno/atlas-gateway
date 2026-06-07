"""OpenAI-compatible request/response schema for /v1/chat/completions.

This is the external contract clients see (the OpenAI shape) for both the
non-streaming response and the streaming `chat.completion.chunk` deltas. It is
distinct from the provider-internal types in `app.providers.base`. See
atlas-docs/03.
"""

from __future__ import annotations

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False


class ResponseMessage(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str = "stop"


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    created: int
    model: str
    choices: list[Choice]
    usage: CompletionUsage
    object: str = "chat.completion"


# --- Streaming (chat.completion.chunk) ---


class ChoiceDelta(BaseModel):
    role: str | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: ChoiceDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    created: int
    model: str
    choices: list[ChunkChoice]
    object: str = "chat.completion.chunk"
    usage: CompletionUsage | None = None

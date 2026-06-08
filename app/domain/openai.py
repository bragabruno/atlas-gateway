"""OpenAI-compatible request/response schema for the gateway endpoints.

This is the external contract clients see (the OpenAI shape) for chat
completions (non-streaming response + streaming `chat.completion.chunk`
deltas), the `/v1/models` list (GW-8), and `/v1/embeddings` (GW-8). It is the
wire DTO of the domain layer, distinct from the provider-internal types in
`app.domain.messages`. See atlas-docs/03 + ADR-016.
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
    #: REG-4 — optional prompt-registry reference (``<name>@<semver|production>``)
    #: and its template params. When present and a prompt registry is wired, the
    #: gateway resolves the ref, renders it, and injects it as a leading system
    #: message. ``None`` (the default) leaves the request unchanged, so a plain
    #: chat request behaves exactly as before.
    prompt_ref: str | None = None
    prompt_params: dict[str, object] | None = None


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


# --- Models list (GET /v1/models, GW-8) ---


class Model(BaseModel):
    id: str
    owned_by: str
    object: str = "model"


class ModelList(BaseModel):
    data: list[Model]
    object: str = "list"


# --- Embeddings (POST /v1/embeddings, GW-8) ---


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]


class EmbeddingData(BaseModel):
    index: int
    embedding: list[float]
    object: str = "embedding"


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage
    object: str = "list"

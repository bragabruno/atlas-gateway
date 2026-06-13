"""OpenAI-compatible request/response schema for the gateway endpoints.

This is the external contract clients see (the OpenAI shape) for chat
completions (non-streaming response + streaming `chat.completion.chunk`
deltas), the `/v1/models` list (GW-8), and `/v1/embeddings` (GW-8). It is the
wire DTO of the domain layer, distinct from the provider-internal types in
`app.domain.messages`. See atlas-docs/03 + ADR-016.

`Field(..., description=...)` and `json_schema_extra={"example": ...}` are
populated so the exported OpenAPI spec (GW-21) is documentation-grade — the
TS frontend client (FE-3) and any Python codegen consumers pick the
descriptions up as TSDoc / docstrings.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"role": "user", "content": "Hello, Atlas."}}
    )

    role: str = Field(
        description="Conversation role. One of `system`, `user`, `assistant`, `tool`.",
    )
    content: str = Field(description="Message content (plain text).")


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "mock",
                "messages": [{"role": "user", "content": "Hello, Atlas."}],
                "max_tokens": 64,
                "temperature": 0.2,
                "stream": False,
            }
        }
    )

    model: str = Field(
        description=("Provider model id or Atlas alias (resolved via `app.providers.registry`)."),
    )
    messages: list[ChatMessage] = Field(description="Ordered conversation messages.")
    max_tokens: int | None = Field(
        default=None,
        description="Cap on generated tokens. `None` defers to the provider default.",
    )
    temperature: float | None = Field(
        default=None,
        description="Sampling temperature in `[0, 2]`. `None` defers to the provider default.",
    )
    stream: bool = Field(
        default=False,
        description=(
            "When `true`, response is `text/event-stream` carrying "
            "`chat.completion.chunk` deltas terminated by `data: [DONE]`."
        ),
    )
    #: REG-4 — optional prompt-registry reference (``<name>@<semver|production>``)
    #: and its template params. When present and a prompt registry is wired, the
    #: gateway resolves the ref, renders it, and injects it as a leading system
    #: message. ``None`` (the default) leaves the request unchanged, so a plain
    #: chat request behaves exactly as before.
    prompt_ref: str | None = Field(
        default=None,
        description=(
            "Optional prompt-registry reference `<name>@<semver|production>` "
            "(REG-4). Resolved + rendered server-side and prepended as a system "
            "message. `None` (default) leaves the request unchanged."
        ),
    )
    prompt_params: dict[str, object] | None = Field(
        default=None,
        description="Template params for `prompt_ref` rendering.",
    )


class ResponseMessage(BaseModel):
    role: str = Field(default="assistant", description="Always `assistant` for responses.")
    content: str = Field(description="Generated message content.")


class Choice(BaseModel):
    index: int = Field(default=0, description="Zero-based choice index.")
    message: ResponseMessage = Field(description="The generated assistant message.")
    finish_reason: str = Field(
        default="stop",
        description="Why generation stopped — `stop`, `length`, `content_filter`, `tool_calls`.",
    )


class CompletionUsage(BaseModel):
    prompt_tokens: int = Field(description="Tokens in the prompt (post-template-render).")
    completion_tokens: int = Field(description="Tokens generated in the completion.")
    total_tokens: int = Field(description="`prompt_tokens + completion_tokens`.")


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "chatcmpl-mock-1",
                "created": 1735689600,
                "model": "mock",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi! How can I help?"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 6, "completion_tokens": 7, "total_tokens": 13},
            }
        }
    )

    id: str = Field(description="Server-assigned completion id.")
    created: int = Field(description="Unix epoch seconds when the response was created.")
    model: str = Field(description="The model that served the request.")
    choices: list[Choice] = Field(description="Generated choices (length ≥ 1).")
    usage: CompletionUsage = Field(description="Token usage for billing + accounting.")
    object: str = Field(default="chat.completion", description="Always `chat.completion`.")


# --- Streaming (chat.completion.chunk) ---


class ChoiceDelta(BaseModel):
    role: str | None = Field(default=None, description="Role on the first chunk only.")
    content: str | None = Field(default=None, description="Token(s) appended this chunk.")


class ChunkChoice(BaseModel):
    index: int = Field(default=0, description="Zero-based choice index (matches non-stream).")
    delta: ChoiceDelta = Field(description="Incremental update for this choice.")
    finish_reason: str | None = Field(
        default=None,
        description="Set on the terminal chunk; `None` on every other chunk.",
    )


class ChatCompletionChunk(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "chatcmpl-mock-1",
                "created": 1735689600,
                "model": "mock",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
            }
        }
    )

    id: str = Field(description="Matches the parent completion id across chunks.")
    created: int = Field(description="Unix epoch seconds for this chunk.")
    model: str = Field(description="The model that served the request.")
    choices: list[ChunkChoice] = Field(description="Per-choice deltas.")
    object: str = Field(
        default="chat.completion.chunk", description="Always `chat.completion.chunk`."
    )
    usage: CompletionUsage | None = Field(
        default=None, description="Only set on the terminal chunk."
    )


# --- Models list (GET /v1/models, GW-8) ---


class Model(BaseModel):
    id: str = Field(description="Model id or Atlas alias.")
    owned_by: str = Field(description="Provider — `openai`, `anthropic`, `google`, `atlas`.")
    object: str = Field(default="model", description="Always `model`.")


class ModelList(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "object": "list",
                "data": [
                    {"id": "mock", "owned_by": "atlas", "object": "model"},
                    {"id": "gpt-4o-mini", "owned_by": "openai", "object": "model"},
                ],
            }
        }
    )

    data: list[Model] = Field(description="All models advertised by the provider registry.")
    object: str = Field(default="list", description="Always `list`.")


# --- Embeddings (POST /v1/embeddings, GW-8) ---


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"model": "mock", "input": "Hello, Atlas."}}
    )

    model: str = Field(description="Provider model id or Atlas alias.")
    input: str | list[str] = Field(description="One string or a batch of strings to embed.")


class EmbeddingData(BaseModel):
    index: int = Field(description="Position in the input batch.")
    embedding: list[float] = Field(description="Dense vector for this input.")
    object: str = Field(default="embedding", description="Always `embedding`.")


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = Field(description="Tokens consumed across the batch.")
    total_tokens: int = Field(description="`prompt_tokens` (no generation for embeddings).")


class EmbeddingResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "object": "list",
                "model": "mock",
                "data": [
                    {"index": 0, "embedding": [0.01, -0.02, 0.03], "object": "embedding"},
                ],
                "usage": {"prompt_tokens": 6, "total_tokens": 6},
            }
        }
    )

    data: list[EmbeddingData] = Field(description="One entry per input, in input order.")
    model: str = Field(description="The model that served the request.")
    usage: EmbeddingUsage = Field(description="Token usage for billing + accounting.")
    object: str = Field(default="list", description="Always `list`.")


# --- Errors (typed envelope for 4xx/5xx) ---


class ErrorEnvelope(BaseModel):
    """Typed error body for non-2xx responses.

    FastAPI's default HTTPException emits `{"detail": ...}`. Some paths (rate
    limit, budget) return a richer body — this envelope is the documented
    superset so codegen produces a single typed error model.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"detail": "invalid api key", "code": "unauthorized"},
        }
    )

    detail: str | dict[str, object] = Field(
        description=(
            "Human-readable error message, or a structured object for guardrail / "
            "rate-limit rejections (see per-endpoint response docs)."
        ),
    )
    code: str | None = Field(
        default=None,
        description="Optional machine-readable error code (provider/guardrail name).",
    )

"""Typed boundary over the OTel GenAI semantic-convention attribute names.

The canonical `gen_ai.*` attribute keys live in
``opentelemetry.semconv._incubating.attributes.gen_ai_attributes`` — a *private*
(`_incubating`) submodule that pyright strict flags as missing a type stub even
though the constants themselves are typed `Final[str]` literals. Rather than
scatter ``# type: ignore`` across the span helpers, this module pins each
attribute key the gateway uses to an explicit ``str``, confining the single
suppression here (the same idiom as `app.limits._redis_typing`). The values are
re-exported from the semconv package, so they track the spec — this file owns
only their *types*, not their literal strings. See atlas-docs/04 §6.2.
"""

from __future__ import annotations

from typing import Final

from opentelemetry.semconv._incubating.attributes import (  # type: ignore[reportMissingTypeStubs]
    gen_ai_attributes as _genai,
)

GEN_AI_OPERATION_NAME: Final[str] = _genai.GEN_AI_OPERATION_NAME
GEN_AI_SYSTEM: Final[str] = _genai.GEN_AI_SYSTEM
GEN_AI_REQUEST_MODEL: Final[str] = _genai.GEN_AI_REQUEST_MODEL
GEN_AI_REQUEST_MAX_TOKENS: Final[str] = _genai.GEN_AI_REQUEST_MAX_TOKENS
GEN_AI_REQUEST_TEMPERATURE: Final[str] = _genai.GEN_AI_REQUEST_TEMPERATURE
GEN_AI_RESPONSE_MODEL: Final[str] = _genai.GEN_AI_RESPONSE_MODEL
GEN_AI_RESPONSE_FINISH_REASONS: Final[str] = _genai.GEN_AI_RESPONSE_FINISH_REASONS
GEN_AI_USAGE_INPUT_TOKENS: Final[str] = _genai.GEN_AI_USAGE_INPUT_TOKENS
GEN_AI_USAGE_OUTPUT_TOKENS: Final[str] = _genai.GEN_AI_USAGE_OUTPUT_TOKENS
GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS: Final[str] = (
    _genai.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS
)
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: Final[str] = _genai.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS

"""Chat use-case orchestration (service layer).

Resolves the provider for a request, drives the provider call, and maps
provider-internal results to the OpenAI-compatible wire schema. The controller
(`app.api.v1.chat`) stays thin — all chat business logic lives here.

Capability collaborators (caching GW-13, rate-limit GW-16, budget GW-17,
accounting GW-14/15, alias routing GW-10, resilience GW-11/12, guardrails
GRD-1/2/4/7/8/9/10, prompt registry REG-3/4) are **injected and optional**, all
defaulting to ``None``. When a collaborator is ``None`` its stage is skipped, so
an unconfigured `ChatService(registry)` behaves byte-for-byte as it did before
this wiring landed — the default/test path resolves a model on the registry and
calls the provider with no extra hops. The composition root (`app.api.deps`)
constructs each collaborator only when its backing config is present (a Redis
URL, a feature flag, a DB pool); absent that, it stays ``None`` and inert. See
ADR-016.

Request-path order (each stage applies only when its collaborator is present):

1. rate-limit (GW-16)            → ``RateLimitExceeded`` (429)
2. budget admission (GW-17)      → ``BudgetExceeded`` (429)
3. pre-guardrails (GRD-1/2/4/7)  → ``GuardrailRejection`` (422); PII redaction
   mutates the messages in place so the provider never sees raw PII
4. prompt_ref resolve (REG-4)    → render a registry prompt into a system message
5. exact-cache get (GW-13)       → a hit returns the cached response immediately
6. on miss: alias route (GW-10) + circuit-breaker/retry (GW-11/12) around the
   provider call
7. accounting (GW-14/15)         → persist a `call_records` row + Kafka event;
   never fails the request (errors are swallowed by the adapters)
8. post-guardrails (GRD-8/9/10)  → ``GuardrailRejection`` (422)
9. cache set (GW-13)             → store the fresh response for later hits

The streaming path applies stages 1–4 and the routing of stage 6, then streams
the provider deltas directly. Post-guardrails (8) and cache set/get (5, 9) are
**deliberately skipped on the streamed path**: they operate on a complete
response body, which does not exist until the stream finishes — buffering the
whole stream to validate/cache it would defeat streaming's first-token latency,
so SSE responses bypass response-shaped stages by design (a later ticket can add
streaming-aware guardrails that inspect the assembled transcript).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from app.cache.exact import NO_PROMPT_VERSION
from app.domain.errors import UnknownModelError
from app.domain.messages import ChatResult, Message, Usage
from app.domain.openai import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    CompletionUsage,
    ResponseMessage,
)
from app.guardrails.chain import GuardrailContext
from app.providers.base import Provider
from app.providers.registry import ProviderRegistry

#: Sentinel tenant id used when no authenticated key is threaded through (the
#: unconfigured default path never consults a tenant-scoped collaborator, so the
#: value is inert there; the wired path always supplies the real key).
_ANON_TENANT = "anonymous"


@dataclass(frozen=True, slots=True)
class CallContext:
    """Realized-call facts handed to the accounting seam (GW-14/15).

    A small value object decoupling the service from the recorder's concrete
    `CallRecord` (which needs DB ids and pinned prices the composition-root
    adapter supplies). The composition-root recorder adapter maps this onto a
    priced `CallRecord` (persisted via GW-14) and an `atlas.calls.v1` event
    (published via GW-15); the fake recorder in tests just captures it.
    """

    api_key_id: str
    model: str
    usage: Usage
    #: Resolved prompt version (REG-4) when the request carried a `prompt_ref`;
    #: ``NO_PROMPT_VERSION`` otherwise. The composition-root adapter maps this to
    #: `CallRecord.prompt_version_id` (None for the sentinel) so accounting rows
    #: attribute spend to the prompt version that produced them.
    prompt_version: str = NO_PROMPT_VERSION


class RateLimiter(Protocol):
    """Port for the per-key token-bucket limiter (GW-16).

    `app.limits.ratelimit.TokenBucketRateLimiter` satisfies this structurally;
    on exhaustion `check` raises `RateLimitExceeded` (mapped to 429). Tests
    inject a fake.
    """

    async def check(self, *, api_key_id: str, alias: str) -> None: ...


class BudgetEnforcer(Protocol):
    """Port for the monthly per-key budget enforcer (GW-17).

    `app.limits.budget.MonthlyBudgetEnforcer` satisfies this structurally.
    `check` is the pre-call admission (raises `BudgetExceeded` → 429 when already
    over cap); `charge` reconciles the realized post-call cost without enforcing.
    """

    async def check(self, *, api_key_id: str, cost: Decimal) -> object: ...

    async def charge(self, *, api_key_id: str, cost: Decimal) -> object: ...


class ResponseCache(Protocol):
    """Port for the exact-match response cache (GW-13).

    `app.cache.exact.ExactCache` satisfies this structurally. Values are opaque
    serialized response strings; the service owns serialization.
    """

    async def get(
        self,
        *,
        tenant_id: str,
        model: str,
        messages: Sequence[Message],
        prompt_version: str,
    ) -> str | None: ...

    async def set(
        self,
        *,
        tenant_id: str,
        model: str,
        messages: Sequence[Message],
        value: str,
        prompt_version: str,
    ) -> None: ...


class GuardrailRunner(Protocol):
    """Port for the pre/post guardrail chain (GRD-1).

    `app.guardrails.chain.GuardrailChain` satisfies this; a rejection raises
    `GuardrailRejection` (mapped to 422).
    """

    async def run_pre(self, ctx: GuardrailContext) -> None: ...

    async def run_post(self, ctx: GuardrailContext) -> None: ...


class RouteResolver(Protocol):
    """Port for the alias routing resolver (GW-10).

    `app.routing.aliases.AliasResolver` satisfies this; `resolve` raises
    `UnknownAliasError` on an unknown alias. The returned object carries
    `provider`, `primary_model`, and `fallback_model`.
    """

    def resolve(
        self,
        alias: str,
        per_key_overrides: Mapping[str, object] | None = None,
    ) -> _RouteTarget: ...


class _RouteTarget(Protocol):
    """Structural view of a routing target (`app.routing.aliases.RouteTarget`)."""

    @property
    def provider(self) -> str: ...

    @property
    def primary_model(self) -> str: ...

    @property
    def fallback_model(self) -> str: ...


class Breaker(Protocol):
    """Port for the per-provider circuit breaker (GW-12).

    `app.resilience.circuit_breaker.CircuitBreaker` satisfies this. `choose`
    picks a healthy provider id (raising `AllProvidersUnavailable` when none is),
    `allow` admits a call (raising `CircuitOpenError` when OPEN), and the
    record_* methods report the outcome so the breaker tracks provider health.
    """

    async def choose(self, primary: str, fallback: str) -> str: ...

    async def allow(self, provider: str) -> None: ...

    async def record_success(self, provider: str) -> None: ...

    async def record_failure(self, provider: str) -> None: ...


class Recorder(Protocol):
    """Port for the accounting recorder + Kafka event sink (GW-14/15).

    Both `app.accounting.recorder.CallRecorder.record` and
    `app.accounting.events.EventPublisher.publish_record` are wired behind this
    single seam; the composition root supplies an adapter that prices the
    `CallContext` into a `CallRecord` and fans out to both. Tests inject a fake
    that just captures the `CallContext`. Implementations must never raise on the
    request path — accounting failures are swallowed by the adapters (GW-15).
    """

    async def record(self, call: CallContext) -> None: ...


class PromptRegistry(Protocol):
    """Port for the prompt registry resolver (REG-3/4).

    `app.registry.resolver.PromptResolver` satisfies this; `resolve` renders a
    `prompt_ref` into a `ResolvedPrompt` (carrying `rendered` and
    `prompt_version_id`).
    """

    def resolve(self, ref: str, params: dict[str, object] | None = None) -> _ResolvedPrompt: ...


class _ResolvedPrompt(Protocol):
    """Structural view of a resolved prompt (`app.registry.resolver.ResolvedPrompt`)."""

    @property
    def prompt_version_id(self) -> str: ...

    @property
    def rendered(self) -> str: ...


class ChatService:
    """Orchestrates chat completions over the provider registry.

    All capability collaborators are optional and default to ``None``; an
    unconfigured `ChatService(registry)` is byte-for-byte the pre-wiring service.
    """

    def __init__(
        self,
        registry: ProviderRegistry,
        *,
        guardrails: GuardrailRunner | None = None,
        cache: ResponseCache | None = None,
        rate_limiter: RateLimiter | None = None,
        budget: BudgetEnforcer | None = None,
        recorder: Recorder | None = None,
        router: RouteResolver | None = None,
        breaker: Breaker | None = None,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self._registry = registry
        self._guardrails = guardrails
        self._cache = cache
        self._rate_limiter = rate_limiter
        self._budget = budget
        self._recorder = recorder
        self._router = router
        self._breaker = breaker
        self._prompt_registry = prompt_registry

    def _resolve(self, model: str) -> Provider:
        provider = self._registry.resolve(model)
        if provider is None:
            raise UnknownModelError(model)
        return provider

    @staticmethod
    def _provider_messages(req: ChatCompletionRequest) -> list[Message]:
        return [Message(role=m.role, content=m.content) for m in req.messages]

    async def _pre_checks(
        self,
        req: ChatCompletionRequest,
        messages: list[Message],
        *,
        api_key_id: str,
    ) -> tuple[list[Message], str]:
        """Run rate-limit, budget, pre-guardrails, and prompt resolution.

        Returns the (possibly redacted / prompt-augmented) messages and the
        prompt-version segment for the cache key. Shared by the streaming and
        non-streaming paths so both enforce the same admission rules. Each stage
        is skipped when its collaborator is ``None``.
        """
        # (1) rate-limit — per-key token bucket → RateLimitExceeded (429).
        if self._rate_limiter is not None:
            await self._rate_limiter.check(api_key_id=api_key_id, alias=req.model)

        # (2) budget admission — pre-call, against already-accrued spend. A zero
        # cost only probes the cap (charge of the realized cost happens post-call
        # in `_record`); raises BudgetExceeded (429) when already at/over cap.
        if self._budget is not None:
            await self._budget.check(api_key_id=api_key_id, cost=Decimal(0))

        # (3) pre-guardrails — size/injection/PII. PII redaction rewrites the
        # context messages so the provider never sees raw PII; injection/size
        # reject fail-fast with GuardrailRejection (422).
        if self._guardrails is not None:
            ctx = GuardrailContext(
                tenant_id=api_key_id,
                model=req.model,
                messages=messages,
            )
            await self._guardrails.run_pre(ctx)
            messages = list(ctx.messages)

        # (4) prompt_ref resolve (REG-4) — when the request carries a prompt_ref,
        # render the registry prompt and inject it as a leading system message.
        prompt_version = NO_PROMPT_VERSION
        if self._prompt_registry is not None and req.prompt_ref is not None:
            resolved = self._prompt_registry.resolve(req.prompt_ref, req.prompt_params)
            prompt_version = resolved.prompt_version_id
            messages = [Message(role="system", content=resolved.rendered), *messages]

        return messages, prompt_version

    async def _call_provider(
        self,
        req: ChatCompletionRequest,
        messages: list[Message],
    ) -> ChatResult:
        """Resolve the target and call the provider with breaker + retry (GW-11/12).

        Routing (GW-10) picks the model/provider when a router is wired; the
        circuit breaker (GW-12) shields the call and records the outcome. When
        neither is wired the model resolves straight on the registry, exactly as
        the pre-wiring service did.
        """
        model = req.model
        provider_id: str | None = None

        if self._router is not None:
            target = self._router.resolve(req.model)
            model = target.primary_model
            provider_id = target.provider
            if self._breaker is not None:
                chosen = await self._breaker.choose(target.provider, target.fallback_model)
                # `choose` returns the provider id to use; on failover the
                # fallback model is the one to call.
                if chosen != target.provider:
                    model = target.fallback_model
                provider_id = chosen

        provider = self._resolve(model)

        if self._breaker is not None and provider_id is not None:
            await self._breaker.allow(provider_id)
            try:
                result = await provider.chat(
                    model=model,
                    messages=messages,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                )
            except Exception:
                await self._breaker.record_failure(provider_id)
                raise
            await self._breaker.record_success(provider_id)
            return result

        return await provider.chat(
            model=model,
            messages=messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )

    async def complete(
        self,
        req: ChatCompletionRequest,
        *,
        api_key_id: str = _ANON_TENANT,
    ) -> ChatCompletionResponse:
        """Run a non-streaming completion and return the OpenAI-shaped response.

        `api_key_id` identifies the caller for the tenant-scoped collaborators
        (rate-limit, budget, cache, accounting); it defaults to a sentinel so the
        unconfigured path (no such collaborators) is unaffected.
        """
        base_messages = self._provider_messages(req)
        messages, prompt_version = await self._pre_checks(req, base_messages, api_key_id=api_key_id)

        # (5) exact-cache get — a hit short-circuits the provider call.
        if self._cache is not None:
            cached = await self._cache.get(
                tenant_id=api_key_id,
                model=req.model,
                messages=messages,
                prompt_version=prompt_version,
            )
            if cached is not None:
                return ChatCompletionResponse.model_validate_json(cached)

        # (6) provider call with routing + breaker/retry.
        result = await self._call_provider(req, messages)
        response = self._to_response(result)

        # (7) accounting record + Kafka event (never fails the request). The
        # resolved prompt_version (REG-4) is threaded through so the row
        # attributes spend to the prompt version that produced the response.
        if self._recorder is not None:
            await self._record(result, api_key_id=api_key_id, prompt_version=prompt_version)

        # (8) post-guardrails — schema/content/citation over the response.
        if self._guardrails is not None:
            post_ctx = GuardrailContext(
                tenant_id=api_key_id,
                model=result.model,
                messages=messages,
                result=result,
            )
            await self._guardrails.run_post(post_ctx)

        # (9) cache set — store the fresh response for later hits.
        if self._cache is not None:
            await self._cache.set(
                tenant_id=api_key_id,
                model=req.model,
                messages=messages,
                value=response.model_dump_json(),
                prompt_version=prompt_version,
            )

        return response

    @staticmethod
    def _to_response(result: ChatResult) -> ChatCompletionResponse:
        usage = CompletionUsage(
            prompt_tokens=result.usage.input_tokens,
            completion_tokens=result.usage.output_tokens,
            total_tokens=result.usage.input_tokens + result.usage.output_tokens,
        )
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=result.model,
            choices=[
                Choice(
                    index=0,
                    message=ResponseMessage(content=result.content),
                    finish_reason=result.finish_reason,
                )
            ],
            usage=usage,
        )

    async def _record(self, result: ChatResult, *, api_key_id: str, prompt_version: str) -> None:
        """Hand the call to the accounting seam (GW-14/15).

        The recorder owns pricing/persistence/Kafka and swallows its own errors,
        so accounting can never fail or stall the user's completion. The seam is
        passed the realized `Usage` + resolved model + `prompt_version` (REG-4)
        so the adapter can price and emit the `call_records` row (with its
        `prompt_version_id`) and `atlas.calls.v1` event. Budget spend is
        reconciled from the same realized cost when a budget enforcer is wired.
        """
        recorder = self._recorder
        if recorder is None:
            return
        await recorder.record(
            CallContext(
                api_key_id=api_key_id,
                model=result.model,
                usage=result.usage,
                prompt_version=prompt_version,
            )
        )

    def stream(
        self,
        req: ChatCompletionRequest,
        *,
        api_key_id: str = _ANON_TENANT,
    ) -> AsyncIterator[str]:
        """Return an SSE frame iterator for a streaming completion.

        The provider is resolved eagerly (raising `UnknownModelError` before the
        response body starts) so the controller can map it to a 404; the
        returned async generator yields the OpenAI-compatible `data:` frames.
        Pre-checks (rate-limit, budget, pre-guardrails, prompt resolution) run
        before the first frame; post-guardrails and cache get/set are skipped on
        the streamed path (see the module docstring).
        """
        provider = self._resolve(req.model)
        return self._stream_frames(req, provider, api_key_id=api_key_id)

    async def _stream_frames(
        self,
        req: ChatCompletionRequest,
        provider: Provider,
        *,
        api_key_id: str,
    ) -> AsyncIterator[str]:
        base_messages = self._provider_messages(req)
        messages, _ = await self._pre_checks(req, base_messages, api_key_id=api_key_id)

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        role_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=req.model,
            choices=[ChunkChoice(delta=ChoiceDelta(role="assistant"))],
        )
        yield f"data: {role_chunk.model_dump_json()}\n\n"

        async for delta in provider.chat_stream(
            model=req.model,
            messages=messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        ):
            if delta.finish_reason is not None or delta.usage is not None:
                usage = (
                    CompletionUsage(
                        prompt_tokens=delta.usage.input_tokens,
                        completion_tokens=delta.usage.output_tokens,
                        total_tokens=delta.usage.input_tokens + delta.usage.output_tokens,
                    )
                    if delta.usage is not None
                    else None
                )
                final_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=req.model,
                    choices=[
                        ChunkChoice(
                            delta=ChoiceDelta(), finish_reason=delta.finish_reason or "stop"
                        )
                    ],
                    usage=usage,
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
            else:
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=req.model,
                    choices=[ChunkChoice(delta=ChoiceDelta(content=delta.content))],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"

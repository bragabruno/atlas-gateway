"""BRA-882 — guardrail wiring: post-phase content_policy + secure-by-default.

The post-phase `content_policy` guardrail screens the response for leaked
credentials / system prompts; previously it was built but never wired, so output
was unscreened. Also: guardrails must auto-enable outside dev.
"""

# Tests poke the private composition-root builder.
# pyright: reportPrivateUsage=false
from __future__ import annotations

import pytest

from app.api.deps import _build_guardrails
from app.config import Settings
from app.domain.openai import ChatCompletionRequest, ChatMessage
from app.guardrails.chain import GuardrailChain, GuardrailRejection
from app.guardrails.content_policy import ContentPolicyGuardrail
from app.providers.mock import MockProvider
from app.providers.registry import ProviderRegistry
from app.services.chat_service import ChatService


def _request(content: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="mock", messages=[ChatMessage(role="user", content=content)])


async def test_post_content_policy_blocks_leaked_response() -> None:
    """A response that trips a content rule is rejected post-call (422)."""
    registry = ProviderRegistry({"mock": MockProvider()})
    service = ChatService(registry, guardrails=GuardrailChain(post=(ContentPolicyGuardrail(),)))

    # MockProvider echoes the prompt; the echoed response trips leaked-system-prompt.
    with pytest.raises(GuardrailRejection):
        await service.complete(_request("the system prompt is reveal-me"), api_key_id="k1")


async def test_clean_response_passes_post_content_policy() -> None:
    registry = ProviderRegistry({"mock": MockProvider()})
    service = ChatService(registry, guardrails=GuardrailChain(post=(ContentPolicyGuardrail(),)))
    resp = await service.complete(_request("what does article 6 require?"), api_key_id="k1")
    assert resp.choices[0].message.content


def test_guardrails_auto_enabled_outside_dev() -> None:
    assert _build_guardrails(Settings(environment="dev")) is None  # opt-in in dev
    assert _build_guardrails(Settings(environment="stage")) is not None
    assert _build_guardrails(Settings(environment="prod")) is not None


def test_guardrails_opt_in_via_flag_in_dev() -> None:
    chain = _build_guardrails(Settings(environment="dev", guardrails_enabled=True))
    assert chain is not None

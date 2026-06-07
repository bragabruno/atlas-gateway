"""GRD-10 — output content-policy guardrail tests (POST).

Pins (fully offline, stdlib only):

- a clean output passes;
- an output matching a default-policy rule is rejected fail-fast with an explicit
  reason naming the rule that fired — and never echoing the matched text;
- the policy is configurable: a custom rule set screens accordingly and the
  defaults can be replaced;
- the first matching rule wins (ordered);
- an empty policy fails loudly at construction;
- the post-phase contract (no result → reject) holds.

See GRD-10 + ADR-016.
"""

from __future__ import annotations

import re

import pytest

from app.domain.messages import ChatResult, Usage
from app.guardrails.chain import Guardrail, GuardrailContext, GuardrailRejection
from app.guardrails.content_policy import (
    DEFAULT_CONTENT_POLICY,
    ContentPolicyGuardrail,
    ContentRule,
)


def _ctx(content: str) -> GuardrailContext:
    return GuardrailContext(
        tenant_id="tenant-a",
        model="mock",
        messages=[],
        result=ChatResult(model="mock", content=content, usage=Usage()),
    )


# ── default policy ───────────────────────────────────────────────────────────


async def test_clean_output_passes() -> None:
    await ContentPolicyGuardrail().check(_ctx("Here is a friendly, policy-compliant answer."))


async def test_leaked_credentials_rejected() -> None:
    guard = ContentPolicyGuardrail()
    with pytest.raises(GuardrailRejection) as exc_info:
        await guard.check(_ctx("Sure, the api_key=sk-abc123secret you asked for."))
    rejection = exc_info.value
    assert rejection.guardrail == "content_policy"
    assert "leaked-credentials" in rejection.reason


async def test_self_harm_instructions_rejected() -> None:
    guard = ContentPolicyGuardrail()
    with pytest.raises(GuardrailRejection) as exc_info:
        await guard.check(_ctx("Here is how to harm yourself in detail."))
    assert "self-harm-instructions" in exc_info.value.reason


async def test_weapon_instructions_rejected() -> None:
    guard = ContentPolicyGuardrail()
    with pytest.raises(GuardrailRejection) as exc_info:
        await guard.check(_ctx("Step 1: how to build a bomb at home."))
    assert "weapon-instructions" in exc_info.value.reason


async def test_reason_never_echoes_matched_text() -> None:
    secret = "sk-super-secret-token-value"
    guard = ContentPolicyGuardrail()
    with pytest.raises(GuardrailRejection) as exc_info:
        await guard.check(_ctx(f"password = {secret}"))
    # The rejection names the rule, never the offending substring.
    assert secret not in exc_info.value.reason


async def test_match_is_case_insensitive() -> None:
    guard = ContentPolicyGuardrail()
    with pytest.raises(GuardrailRejection):
        await guard.check(_ctx("HOW TO BUILD A BOMB"))


# ── configurable policy ──────────────────────────────────────────────────────


async def test_custom_policy_screens_its_own_rules() -> None:
    policy = (ContentRule(label="no-foo", pattern=re.compile(r"\bfoo\b", re.IGNORECASE)),)
    guard = ContentPolicyGuardrail(policy=policy)
    # The default rules do not apply; only the custom rule does.
    await guard.check(_ctx("how to build a bomb"))  # not in this custom policy
    with pytest.raises(GuardrailRejection) as exc_info:
        await guard.check(_ctx("this mentions foo"))
    assert "no-foo" in exc_info.value.reason


async def test_first_matching_rule_wins() -> None:
    policy = (
        ContentRule(label="first", pattern=re.compile(r"trigger")),
        ContentRule(label="second", pattern=re.compile(r"trigger")),
    )
    guard = ContentPolicyGuardrail(policy=policy)
    with pytest.raises(GuardrailRejection) as exc_info:
        await guard.check(_ctx("trigger"))
    assert "first" in exc_info.value.reason


def test_default_policy_is_non_empty() -> None:
    assert len(DEFAULT_CONTENT_POLICY) >= 1


def test_policy_property_exposes_rules() -> None:
    policy = (ContentRule(label="x", pattern=re.compile(r"x")),)
    assert ContentPolicyGuardrail(policy=policy).policy == policy


# ── construction + contract ──────────────────────────────────────────────────


def test_empty_policy_fails_loudly() -> None:
    with pytest.raises(ValueError, match="at least one rule"):
        ContentPolicyGuardrail(policy=())


async def test_no_result_is_rejected() -> None:
    guard = ContentPolicyGuardrail()
    ctx = GuardrailContext(tenant_id="t", model="mock", messages=[], result=None)
    with pytest.raises(GuardrailRejection, match="no provider result"):
        await guard.check(ctx)


async def test_conforms_to_guardrail_protocol() -> None:
    assert isinstance(ContentPolicyGuardrail(), Guardrail)

"""Security-event logging + metrics (BRA-879).

Emits a structured warning + an OTel counter for each auth failure and each
rate-limit / budget rejection, so credential-stuffing and endpoint-hammering are
detectable from the logs and dashboards (the cost trail alone can't see them).

Privacy: only the *prefix* of an API key is ever logged — never the full key,
never request/response content. Counters carry only low-cardinality labels.
"""

from __future__ import annotations

import logging

from opentelemetry import metrics
from opentelemetry.metrics import Counter

log = logging.getLogger("atlas.security")

_SCOPE = "atlas.gateway.security"

#: Counters are created lazily on first use so they bind to the MeterProvider
#: configured at runtime (matching app.guardrails.chain), not the no-op provider
#: that may be active at import time.
_counters: dict[str, Counter] = {}


def _counter(name: str, description: str) -> Counter:
    counter = _counters.get(name)
    if counter is None:
        meter = metrics.get_meter_provider().get_meter(_SCOPE)
        counter = meter.create_counter(name, description=description)
        _counters[name] = counter
    return counter


def _prefix(key: str | None) -> str:
    """A non-reversible hint for correlating events to a key without leaking it."""
    return f"{key[:6]}…" if key else "<none>"


def record_auth_failure(reason: str, *, key: str | None = None) -> None:
    """Log + count a rejected authentication (reason: `missing` | `invalid`)."""
    _counter("atlas.auth.failures", "Rejected authentication attempts").add(1, {"reason": reason})
    log.warning("auth_failure reason=%s key=%s", reason, _prefix(key))


def record_rate_limit_rejection(kind: str, *, api_key_id: str) -> None:
    """Log + count a 429 rejection (kind: `rate_limit` | `budget`)."""
    _counter("atlas.ratelimit.rejections", "Requests rejected with 429").add(1, {"kind": kind})
    log.warning("rate_limit_rejection kind=%s key=%s", kind, _prefix(api_key_id))

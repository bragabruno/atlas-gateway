"""DB-authoritative API-key validation (BRA-881).

When the gateway has a database, the `api_keys` table is the source of truth for
auth: a presented key is valid iff a row with its hash exists, is `active`, and
has not expired. This makes revocation/suspension/expiry take effect at runtime
(set `status='revoked'` or `expires_at` in the DB — no redeploy). The env
allowlist remains the fallback only when no DB is configured (the offline /
test path); that wiring lives in `app.api.deps.require_api_key`.

Only the *hash* of a key is ever stored or compared — never the plaintext.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any, Protocol

#: Prefix tags the scheme so a future migration can rotate it without ambiguity.
_HASH_PREFIX = "sha256:"


def hash_key(key: str) -> str:
    """Stable, non-reversible hash stored in / matched against `hashed_secret`."""
    return _HASH_PREFIX + hashlib.sha256(key.encode("utf-8")).hexdigest()


class _Conn(Protocol):
    """The asyncpg `fetchrow` surface (a Pool or Connection both satisfy it)."""

    async def fetchrow(self, query: str, *args: Any) -> Any: ...


_LOOKUP_SQL = "SELECT status, expires_at FROM api_keys WHERE hashed_secret = $1"


async def key_is_active(conn: _Conn, key: str) -> bool:
    """Return True iff the key exists, is `active`, and is not past `expires_at`."""
    row = await conn.fetchrow(_LOOKUP_SQL, hash_key(key))
    if row is None:
        return False
    if str(row["status"]) != "active":
        return False
    expires_at = row["expires_at"]
    return expires_at is None or expires_at > datetime.now(tz=UTC)

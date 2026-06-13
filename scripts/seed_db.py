#!/usr/bin/env python3
"""Seed the gateway Postgres with dev rows + synthetic call_records.

Stdlib + asyncpg only (no Faker) so it runs unmodified inside the gateway
image:

    docker compose -f local/compose.dev.yaml --profile jobs run --rm seed-db \\
        python scripts/seed_db.py --records 500 --kafka

What it seeds (all idempotent):
- `api_keys`   — one row per --api-key (default: dev-key), id = UUIDv5 of the
  key string in the SAME namespace the accounting adapter uses, so live
  traffic and seeded rows reference the same key id (and the FK on
  call_records holds).
- `model_aliases` — every `ALIAS_SEED` row, plus zero-rate rows for `mock`
  and any --extra-model (e.g. local Ollama ids) so joins/browsing work.
- `call_records`  — --records synthetic rows spread over --days, priced with
  the real `compute_cost`, via the real `CallRecorder` (idempotent inserts).
- `--kafka`       — also publish each synthetic record as an `atlas.calls.v1`
  event (visible in Redpanda Console).

Env: ATLAS_SEED_DSN (default postgresql://atlas:atlas@localhost:5432/atlas),
     ATLAS_KAFKA_BOOTSTRAP (default localhost:9092).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.accounting.adapter import API_KEY_NS, provider_for_model  # noqa: E402
from app.accounting.recorder import CallRecord, CallRecorder, Rates, compute_cost  # noqa: E402
from app.domain.messages import Usage  # noqa: E402
from app.repositories.seed import ALIAS_SEED  # noqa: E402

#: Models the synthetic traffic mixes over (alias or raw model id → rates).
_TRAFFIC_MIX: list[tuple[str, Rates]] = [
    ("mock", Rates(Decimal("0"), Decimal("0"))),
    ("smart", Rates(Decimal("3.00"), Decimal("15.00"))),
    ("fast", Rates(Decimal("1.00"), Decimal("5.00"))),
    ("deep", Rates(Decimal("5.00"), Decimal("25.00"))),
    ("gpt-oss:120b-cloud", Rates(Decimal("0"), Decimal("0"))),
]

_APPS = ["atlas-frontend", "atlas-agent-runtime", "locust", "postman"]


async def _seed_api_keys(conn: object, keys: list[str]) -> None:
    for key in keys:
        await conn.execute(  # type: ignore[attr-defined]
            """
            INSERT INTO api_keys (id, hashed_secret, app, owner, status)
            VALUES ($1, $2, $3, $4, 'active')
            ON CONFLICT (id) DO NOTHING
            """,
            uuid.uuid5(API_KEY_NS, key),
            f"seed-sha256:{key}",  # placeholder — local dev only, never a real secret
            "local-dev",
            "seed_db.py",
        )
    print(f"api_keys: upserted {len(keys)} row(s)")


async def _seed_aliases(conn: object, extra_models: list[str]) -> None:
    rows = [
        (
            r["alias"],
            r["primary_model_id"],
            r["fallback_model_id"],
            r["provider"],
            r["in_price_per_1m"],
            r["out_price_per_1m"],
        )
        for r in ALIAS_SEED
    ]
    for model in ["mock", *extra_models]:
        rows.append((model, model, model, provider_for_model(model).value, Decimal("0"), Decimal("0")))

    for alias, primary, fallback, provider, in_rate, out_rate in rows:
        await conn.execute(  # type: ignore[attr-defined]
            """
            INSERT INTO model_aliases
                (alias, primary_model_id, fallback_model_id, provider,
                 in_price_per_1m, out_price_per_1m, per_key_overrides)
            VALUES ($1, $2, $3, $4, $5, $6, '{}')
            ON CONFLICT (alias) DO NOTHING
            """,
            alias,
            primary,
            fallback,
            provider,
            in_rate,
            out_rate,
        )
    print(f"model_aliases: upserted {len(rows)} row(s)")


def _synthetic_record(api_key: str, *, days: int) -> CallRecord:
    model, rates = random.choice(_TRAFFIC_MIX)
    usage = Usage(
        input_tokens=random.randint(50, 4000),
        output_tokens=random.randint(20, 1500),
        cache_read_input_tokens=random.choice([0, 0, 0, random.randint(100, 2000)]),
    )
    return CallRecord(
        id=uuid.uuid4(),
        api_key_id=uuid.uuid5(API_KEY_NS, api_key),
        app=random.choice(_APPS),
        model=model,
        provider=provider_for_model(model),
        usage=usage,
        cost=compute_cost(usage, rates),
        latency_ms=random.randint(120, 4500),
        status=random.choices([200, 429, 422], weights=[94, 4, 2])[0],
    )


async def _seed_records(
    conn: object, *, records: int, days: int, api_keys: list[str], kafka: str | None
) -> None:
    recorder = CallRecorder(conn)  # type: ignore[arg-type]

    publisher = None
    producer = None
    if kafka:
        from app.accounting.events import AIOKafkaProducerAdapter, EventPublisher

        producer = AIOKafkaProducerAdapter(bootstrap_servers=kafka)
        await producer.start()
        publisher = EventPublisher(producer)

    for i in range(records):
        record = _synthetic_record(random.choice(api_keys), days=days)
        await recorder.record(record)
        # Spread created_at over the window (insert default is now()).
        offset = timedelta(
            days=random.uniform(0, days), seconds=random.uniform(0, 86400)
        )
        await conn.execute(  # type: ignore[attr-defined]
            "UPDATE call_records SET created_at = $1 WHERE id = $2",
            datetime.now(tz=UTC) - offset,
            record.id,
        )
        if publisher is not None:
            await publisher.publish_record(record)
        if (i + 1) % 100 == 0:
            print(f"call_records: {i + 1}/{records}")

    if producer is not None:
        await producer.stop()
    print(f"call_records: inserted {records} row(s)" + (" + kafka events" if kafka else ""))


async def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the gateway DB with dev data")
    parser.add_argument("--records", type=int, default=200, help="synthetic call_records to insert")
    parser.add_argument("--days", type=int, default=14, help="spread records over the last N days")
    parser.add_argument(
        "--api-key", action="append", default=None, help="bearer key(s) to seed (default: dev-key)"
    )
    parser.add_argument(
        "--extra-model",
        action="append",
        default=["gpt-oss:120b-cloud", "kimi-k2.5:cloud"],
        help="extra zero-rate model alias rows",
    )
    parser.add_argument("--kafka", action="store_true", help="also publish atlas.calls.v1 events")
    args = parser.parse_args()

    keys = args.api_key or ["dev-key"]
    dsn = os.environ.get("ATLAS_SEED_DSN", "postgresql://atlas:atlas@localhost:5432/atlas")
    kafka = os.environ.get("ATLAS_KAFKA_BOOTSTRAP", "localhost:9092") if args.kafka else None

    import asyncpg  # type: ignore[import-untyped]

    conn = await asyncpg.connect(dsn)
    try:
        await _seed_api_keys(conn, keys)
        await _seed_aliases(conn, args.extra_model)
        await _seed_records(conn, records=args.records, days=args.days, api_keys=keys, kafka=kafka)
    finally:
        await conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

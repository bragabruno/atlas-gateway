"""GW-21 — OpenAPI drift guard.

Asserts the committed `openapi.json` (the source of truth consumers codegen
against) matches the live `app.openapi()`. If a route, schema, or version
changes without regenerating the spec, this test fails and points the author
at `scripts/export_openapi.py`. Fully offline. See ADR-016.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.main import app

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OPENAPI_PATH = _REPO_ROOT / "openapi.json"


def test_committed_openapi_matches_live_spec() -> None:
    assert _OPENAPI_PATH.is_file(), (
        f"missing {_OPENAPI_PATH}; run `.venv/bin/python scripts/export_openapi.py`"
    )
    committed = json.loads(_OPENAPI_PATH.read_text(encoding="utf-8"))
    live = app.openapi()
    assert committed == live, (
        "openapi.json is out of date; regenerate with `.venv/bin/python scripts/export_openapi.py`"
    )


def test_committed_openapi_has_stable_ordering_and_trailing_newline() -> None:
    raw = _OPENAPI_PATH.read_text(encoding="utf-8")
    assert raw.endswith("\n"), "openapi.json must end with a trailing newline"
    expected = json.dumps(app.openapi(), sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    assert raw == expected, (
        "openapi.json is not in stable (sort_keys) form; regenerate with "
        "`.venv/bin/python scripts/export_openapi.py`"
    )

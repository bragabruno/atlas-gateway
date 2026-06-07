"""Prompt registry runtime (REG-*).

The in-process registry the gateway and agent executor call before an LLM call
to turn a `prompt_ref` into a runnable config (atlas-docs/03 §7.1): `resolver`
chooses the correct `prompt_versions` row by status, validates caller params
against the version's JSON-Schema `params_schema`, and renders the Jinja2
template (REG-3); `promotion` walks a version through the draft → candidate →
production lifecycle and flips the production pointer for instant rollback
(REG-5). A capability module per ADR-016: repositories are injected so the
registry is exercised fully offline, and REG-4 wires `resolve` into the request
path in a later ticket (module-first, wire-later — see ADR-016). Nothing here
imports FastAPI or the HTTP layer.
"""

from __future__ import annotations

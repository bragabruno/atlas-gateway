"""GRD-3 — PII NER guardrail (off-inline path).

Named-entity recognition for novel PII using Presidio + the pinned spaCy model
``en_core_web_lg``.  This is deliberately **NOT** placed on the <50ms inline
request path (see atlas-docs/05 latency strategy): callers wire it as a
background post-processing check or in a separate slow-path chain, not in the
hot ``GuardrailChain`` that runs synchronously during the provider call.

Why a separate ticket from GRD-2 (regex fast-path):
- Regex (GRD-2) catches known-format PII (email, phone, SSN, card numbers) with
  <1 ms overhead on the inline path.
- NER (GRD-3) catches novel or format-less PII (person names, org names, dates of
  birth, addresses, etc.) that regex can't enumerate — but at the cost of ~50–200 ms
  per call for the spaCy inference pass.

Operational requirements (``Done when``):
- NER catches formats regex misses (tested with person-name fixtures).
- Inline p95 budget is preserved — this check MUST NOT run on the inline path.
- Model is pinned: ``en_core_web_lg==3.8.3`` loaded from ``SPACY_MODEL_PATH`` or
  the spaCy default model path so the image doesn't carry a fresh download.

See GRD-2 for the regex fast-path guardrail that covers common PII formats.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from app.domain.messages import Message
from app.guardrails.chain import GuardrailContext, GuardrailPhase, GuardrailRejection

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine

log = logging.getLogger(__name__)

#: Presidio entity types the NER check looks for beyond what GRD-2 covers.
_NER_ENTITIES = [
    "PERSON",
    "LOCATION",
    "DATE_TIME",
    "NRP",           # Nationalities, religious or political groups
    "ORGANIZATION",
]

#: spaCy model name (must be installed in the image; pinned in requirements).
_SPACY_MODEL = "en_core_web_lg"

#: Thread pool for blocking spaCy inference — keeps the asyncio loop free.
_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None


def _get_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="pii-ner"
        )
    return _EXECUTOR


def _load_analyzer() -> "AnalyzerEngine":
    """Lazy-load the Presidio AnalyzerEngine with the spaCy NLP backend."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": _SPACY_MODEL}],
    })
    return AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])


class PiiNerGuardrail:
    """GRD-3 — Presidio NER check for novel PII.

    This guardrail MUST be used off the inline request path; see module docstring.
    Wire it via ``GuardrailChain(post=[PiiNerGuardrail()])`` in a background task,
    or behind a separate slow-path chain.

    The analyzer is instantiated lazily on first use and cached for the lifetime
    of the guardrail instance. spaCy inference runs in a thread-pool to avoid
    blocking the asyncio event loop.
    """

    name = "pii_ner"

    def __init__(self, *, entities: Sequence[str] = _NER_ENTITIES) -> None:
        self._entities = list(entities)
        self._analyzer: AnalyzerEngine | None = None

    def _get_analyzer(self) -> "AnalyzerEngine":
        if self._analyzer is None:
            self._analyzer = _load_analyzer()
        return self._analyzer

    def _analyze_text(self, text: str) -> list[str]:
        """Return detected entity types; runs in a thread."""
        analyzer = self._get_analyzer()
        results = analyzer.analyze(text=text, entities=self._entities, language="en")
        return [r.entity_type for r in results]

    async def check(self, ctx: GuardrailContext) -> None:
        """Run NER on all message content; raise on PII detection.

        Runs the spaCy inference in a thread pool so the asyncio loop stays
        responsive during the potentially long (~50–200 ms) inference.
        """
        loop = asyncio.get_running_loop()
        executor = _get_executor()
        for msg in ctx.messages:
            detected = await loop.run_in_executor(
                executor, self._analyze_text, msg.content
            )
            if detected:
                types = ", ".join(sorted(set(detected)))
                raise GuardrailRejection(
                    guardrail=self.name,
                    phase=GuardrailPhase.PRE,
                    reason=f"NER detected PII entity types: {types}",
                )

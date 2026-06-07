"""GRD-6 — Tool/MCP output sanitization before context re-entry.

Defense against *injection-via-tool-results*: a tool or MCP server can return
attacker-influenced text (a poisoned web page, a planted document chunk, a
crafted API response). If that text re-enters the agent's context verbatim it
becomes an injection vector — "ignore your instructions", a forged
``System:`` turn, or a hidden directive smuggled in a zero-width / control-char
payload. This module neutralizes those payloads *before* the tool output is
turned into a message the model sees.

What "neutralize" means here is **defang, not classify**: the sanitizer does not
decide whether the output is malicious and reject it (that would break legitimate
tool results that merely discuss instructions). Instead it makes the output safe
to re-inject by

1. stripping control / zero-width characters used to hide instructions;
2. defusing forged role turns (a line that starts ``System:`` / ``Assistant:``)
   so the model cannot mistake tool data for a privileged turn;
3. neutralizing the known injection imperatives (``ignore previous
   instructions`` and friends) by wrapping the trigger word so the phrase loses
   its imperative force while the text stays human-readable;
4. fencing the whole result in an explicit untrusted-content delimiter so the
   downstream prompt assembler can keep tool data lexically separated from
   trusted instructions.

The export is reusable on purpose: GRD-6 ships the capability and the agent
runtime (AGT-5) wires :func:`sanitize_tool_output` / :class:`ToolOutputSanitizer`
into the loop before results re-enter context. Stdlib ``re`` / ``unicodedata``
only — no external deps, no network. A capability module per ADR-016: standalone,
fully unit-tested, wired later. Fail-fast on bad input (a non-``str`` payload is
a programming error and raises) — but a *clean* payload is returned defanged, not
rejected, so the agent loop keeps making progress. See GRD-6 + ADR-016.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

#: Default fence labels wrapping a sanitized tool result. The downstream prompt
#: assembler treats everything between these markers as untrusted data, never as
#: instructions. Kept as plain ASCII so they survive any later transport.
_FENCE_OPEN = "[BEGIN UNTRUSTED TOOL OUTPUT]"
_FENCE_CLOSE = "[END UNTRUSTED TOOL OUTPUT]"

#: Unicode "format" (Cf) characters — zero-width space/joiner, bidi overrides,
#: the BOM — are removed outright: they carry no visible meaning but are a known
#: channel for hiding directives inside otherwise-innocent text. Other control
#: characters (category Cc) are removed too, except the whitespace that
#: legitimately structures text (tab, newline, carriage return).
_KEEP_CONTROL = frozenset({"\t", "\n", "\r"})

#: A forged role turn at the start of a line — ``System:``/``Assistant:``/etc.
#: A tool result must never be able to impersonate a privileged conversation
#: turn, so the leading role token is defanged (see ``_NEUTRALIZED_ROLE``).
_FORGED_ROLE_RE = re.compile(
    r"(?im)^[ \t]*(system|assistant|user|developer|tool)[ \t]*:",
)

#: Known injection imperatives. The captured trigger verb is wrapped rather than
#: deleted so the sentence stays readable for debugging/audit while losing its
#: imperative force once fenced. Mirrors the GRD-4 inbound heuristics, applied
#: here to the *outbound-from-tool* direction.
_INJECTION_RES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore-previous-instructions",
        re.compile(r"(?i)\b(ignore|disregard|forget)\b(?=[\s\S]{0,40}\binstructions?\b)"),
    ),
    (
        "override-system-prompt",
        re.compile(r"(?i)\b(override|bypass)\b(?=[\s\S]{0,40}\b(?:system\s+)?prompt\b)"),
    ),
    (
        "reveal-system-prompt",
        re.compile(r"(?i)\b(reveal|leak)\b(?=[\s\S]{0,40}\b(?:system\s+)?prompt\b)"),
    ),
)

#: How a forged leading role token is rewritten: a non-colon separator so it can
#: no longer parse as a role turn, and a visible marker that it was defanged.
_NEUTRALIZED_ROLE = r"(defanged-role: \1)"


def _strip_hidden_chars(text: str) -> str:
    """Drop zero-width/format and control chars used to smuggle directives.

    Removes every Unicode ``Cf`` (format) char and every ``Cc`` (control) char
    except the structural whitespace in `_KEEP_CONTROL`. Visible text is
    untouched, so a clean payload reads identically afterward.
    """
    out: list[str] = []
    for ch in text:
        if ch in _KEEP_CONTROL:
            out.append(ch)
            continue
        category = unicodedata.category(ch)
        if category in ("Cf", "Cc"):
            continue
        out.append(ch)
    return "".join(out)


def _defuse_forged_roles(text: str) -> str:
    """Rewrite leading ``System:``-style role tokens so they cannot impersonate a turn."""
    return _FORGED_ROLE_RE.sub(_NEUTRALIZED_ROLE, text)


def _neutralize_injections(text: str) -> str:
    """Wrap known injection trigger verbs so the imperative loses its force.

    The trigger word (e.g. ``ignore``) is bracketed in place; the surrounding
    text is preserved, so the result stays auditable rather than silently
    rewritten beyond recognition.
    """
    for _label, pattern in _INJECTION_RES:
        text = pattern.sub(r"[\1]", text)
    return text


@dataclass(frozen=True, slots=True)
class SanitizedToolOutput:
    """The result of sanitizing one tool/MCP output.

    ``text`` is the safe-to-re-inject, fenced payload. ``fenced`` is ``True``
    when the untrusted-content delimiters were added (the default). Returned as a
    value object so callers (AGT-5) can log/trace the transform without re-running
    it; equality is by value for easy assertions.
    """

    text: str
    fenced: bool


class ToolOutputSanitizer:
    """Neutralizes tool/MCP output before it re-enters agent context (GRD-6).

    Reusable capability the agent runtime (AGT-5) composes into its loop. The
    transform is deterministic and order-fixed: strip hidden chars → defuse
    forged role turns → neutralize injection imperatives → fence as untrusted.
    ``fence`` may be disabled by a caller that applies its own delimiters, but it
    is on by default because lexical separation of tool data from instructions is
    the primary defense. Fail-fast: a non-``str`` payload raises ``TypeError``
    (a wiring bug), never silently coerced.
    """

    def __init__(self, *, fence: bool = True) -> None:
        self._fence = fence

    @property
    def fences(self) -> bool:
        """Whether this sanitizer wraps output in untrusted-content delimiters."""
        return self._fence

    def sanitize(self, output: object) -> SanitizedToolOutput:
        """Return `output` defanged and (by default) fenced as untrusted content.

        Accepts ``object`` at the boundary and fails fast: a non-``str`` payload
        raises ``TypeError`` (a tool that returned a non-text payload must be
        adapted upstream, not guessed at here) rather than being silently coerced.
        """
        if not isinstance(output, str):
            raise TypeError(f"tool output must be str, got {type(output).__name__}")

        cleaned = _strip_hidden_chars(output)
        cleaned = _defuse_forged_roles(cleaned)
        cleaned = _neutralize_injections(cleaned)

        if self._fence:
            cleaned = f"{_FENCE_OPEN}\n{cleaned}\n{_FENCE_CLOSE}"

        return SanitizedToolOutput(text=cleaned, fenced=self._fence)


def sanitize_tool_output(output: object, *, fence: bool = True) -> str:
    """Module-level convenience: sanitize one tool output and return the text.

    Thin wrapper over :class:`ToolOutputSanitizer` for the common single-call
    case (AGT-5 may instead hold a sanitizer instance). Same fail-fast contract:
    a non-``str`` payload raises ``TypeError``.
    """
    return ToolOutputSanitizer(fence=fence).sanitize(output).text

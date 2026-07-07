"""Guardrails: input/output scanning, PII/secret redaction, injection checks.

Reuses the regex-based detection approach proven in the mcp-audit/envsentry
projects. A ``Guardrail`` inspects text and returns a ``GuardResult`` that can
allow, redact (rewrite), or block. The kernel runs input guardrails on task
payloads before an agent sees them and output guardrails on results before they
are stored — and the same redactor feeds the tracer so secrets never hit logs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- detectors ---------------------------------------------------------------

_SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\bgh[posru]_[A-Za-z0-9]{36,}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "us_ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[ -]?)?(?:\(?\d{3}\)?[ -]?)\d{3}[ -]?\d{4}\b"),
}
# Heuristic prompt-injection markers.
_INJECTION = re.compile(
    r"(?i)(ignore (?:all|previous|above) (?:instructions|prompts)"
    r"|disregard (?:the|your) (?:system|previous)"
    r"|you are now (?:a|an|dan)\b"
    r"|reveal (?:your )?(?:system prompt|instructions))"
)


@dataclass
class GuardResult:
    allowed: bool
    text: str                       # possibly-redacted text
    findings: list[str] = field(default_factory=list)
    reason: str | None = None


def redact(text: str) -> str:
    """Replace any secret/PII match with a typed placeholder like [REDACTED:email]."""
    out = text
    for name, pat in {**_SECRET_PATTERNS, **_PII_PATTERNS}.items():
        out = pat.sub(f"[REDACTED:{name}]", out)
    return out


class Guardrail:
    """Configurable scanner. ``mode`` decides what happens on a hit."""

    def __init__(self, *, block_secrets: bool = True, redact_pii: bool = True,
                 block_injection: bool = True) -> None:
        self.block_secrets = block_secrets
        self.redact_pii = redact_pii
        self.block_injection = block_injection

    def check(self, text: str) -> GuardResult:
        if not isinstance(text, str):
            return GuardResult(allowed=True, text=text)
        findings: list[str] = []

        if self.block_injection and _INJECTION.search(text):
            return GuardResult(allowed=False, text=text, findings=["prompt_injection"],
                               reason="possible prompt injection detected")

        secrets_hit = [n for n, p in _SECRET_PATTERNS.items() if p.search(text)]
        if secrets_hit and self.block_secrets:
            return GuardResult(allowed=False, text=text, findings=secrets_hit,
                               reason=f"blocked: secret material ({', '.join(secrets_hit)})")

        out = text
        pii_hit = [n for n, p in _PII_PATTERNS.items() if p.search(text)]
        if pii_hit and self.redact_pii:
            out = redact(out)
            findings.extend(pii_hit)

        return GuardResult(allowed=True, text=out, findings=findings)

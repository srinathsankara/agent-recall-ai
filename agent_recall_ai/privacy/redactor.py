"""
PIIRedactor — Enterprise privacy middleware for agent-recall-ai.

Scans checkpoint state before serialization and redacts:
  - API keys and tokens (OpenAI, Anthropic, AWS, GitHub, etc.)
  - Passwords and secrets in tool call outputs
  - Email addresses, phone numbers, SSNs (configurable)
  - Credit card numbers and bank account numbers
  - IP addresses in private ranges (configurable)
  - Custom patterns defined by the caller

Design principles:
  1. Scan BEFORE serialization — secrets never hit disk/Redis
  2. Lossless audit trail — redacted items are logged (not the value)
  3. Reversible redaction — placeholder tokens map to originals in-memory
     (so the running agent keeps full context; only the saved copy is redacted)
  4. Zero false negatives on high-severity patterns — regex is conservative
  5. Configurable sensitivity levels — operators choose their compliance posture

Usage:
    from agent_recall_ai.privacy import PIIRedactor, SensitivityLevel
    from agent_recall_ai import Checkpoint

    redactor = PIIRedactor(sensitivity=SensitivityLevel.HIGH)
    with Checkpoint("my-task", redactor=redactor) as cp:
        ...
        # Checkpoint is auto-redacted before every save
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SensitivityLevel(str, Enum):
    """Controls which categories of PII are scanned."""
    LOW = "low"        # Only API keys/tokens/passwords
    MEDIUM = "medium"  # + emails, phone numbers
    HIGH = "high"      # + SSNs, CC numbers, IPs, custom
    PARANOID = "paranoid"  # + any 20+ char alphanumeric that looks like a key


@dataclass
class RedactionRule:
    """A single PII detection pattern."""
    name: str
    pattern: re.Pattern
    placeholder: str        # e.g. "[REDACTED:api_key]"
    severity: SensitivityLevel
    description: str = ""


@dataclass
class RedactionResult:
    """Summary of what was redacted from a checkpoint."""
    session_id: str
    total_redactions: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    fields_touched: list[str] = field(default_factory=list)
    # Maps placeholder → original value (kept in memory only, never persisted)
    _reverse_map: dict[str, str] = field(default_factory=dict, repr=False)

    def add(self, category: str, field_name: str, original: str, placeholder: str) -> None:
        self.total_redactions += 1
        self.by_category[category] = self.by_category.get(category, 0) + 1
        if field_name not in self.fields_touched:
            self.fields_touched.append(field_name)
        self._reverse_map[placeholder] = original

    def restore(self, text: str) -> str:
        """Restore redacted values in text (for in-memory agent use)."""
        for placeholder, original in self._reverse_map.items():
            text = text.replace(placeholder, original)
        return text

    @property
    def was_redacted(self) -> bool:
        return self.total_redactions > 0


# ── Built-in detection patterns ───────────────────────────────────────────────

def _p(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE)


_BUILTIN_RULES: list[RedactionRule] = [

    # ── API Keys & Tokens (always checked) ───────────────────────────────────
    # NOTE: More-specific prefixes (anthropic sk-ant-) must come before the
    # broader OpenAI sk- pattern so the right placeholder is applied.
    RedactionRule(
        name="anthropic_api_key",
        pattern=_p(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
        placeholder="[REDACTED:anthropic_key]",
        severity=SensitivityLevel.LOW,
        description="Anthropic API key",
    ),
    RedactionRule(
        name="openai_api_key",
        pattern=_p(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}"),
        placeholder="[REDACTED:openai_key]",
        severity=SensitivityLevel.LOW,
        description="OpenAI API key",
    ),
    RedactionRule(
        name="aws_access_key",
        pattern=_p(r"AKIA[0-9A-Z]{16}"),
        placeholder="[REDACTED:aws_access_key]",
        severity=SensitivityLevel.LOW,
        description="AWS Access Key ID",
    ),
    RedactionRule(
        name="aws_secret_key",
        pattern=_p(r"(?:aws_secret(?:_access)?_key|AWS_SECRET)[\"'\s:=]+([A-Za-z0-9/+]{40})"),
        placeholder="[REDACTED:aws_secret]",
        severity=SensitivityLevel.LOW,
        description="AWS Secret Access Key",
    ),
    RedactionRule(
        name="github_token",
        pattern=_p(r"gh[pousr]_[A-Za-z0-9_]{36,}"),
        placeholder="[REDACTED:github_token]",
        severity=SensitivityLevel.LOW,
        description="GitHub Personal Access Token",
    ),
    RedactionRule(
        name="bearer_token",
        pattern=_p(r"(?:Bearer|Authorization:\s*Bearer)\s+([A-Za-z0-9\-._~+/]{20,}={0,2})"),
        placeholder="[REDACTED:bearer_token]",
        severity=SensitivityLevel.LOW,
        description="HTTP Bearer token",
    ),
    RedactionRule(
        name="generic_password",
        pattern=_p(r"(?:password|passwd|secret|pwd)[\s\"']*[=:]+[\s\"']*([^\s\"']{8,})"),
        placeholder="[REDACTED:password]",
        severity=SensitivityLevel.LOW,
        description="Generic password assignment",
    ),
    RedactionRule(
        name="private_key_block",
        pattern=_p(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        placeholder="[REDACTED:private_key_header]",
        severity=SensitivityLevel.LOW,
        description="PEM private key block",
    ),
    RedactionRule(
        name="database_url",
        pattern=_p(r"(?:postgres|mysql|mongodb|redis)(?:\+\w+)?://[^@\s]+:[^@\s]+@[^\s]+"),
        placeholder="[REDACTED:database_url]",
        severity=SensitivityLevel.LOW,
        description="Database connection string with credentials",
    ),

    # ── PII: Medium+ ─────────────────────────────────────────────────────────
    RedactionRule(
        name="email_address",
        pattern=_p(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z]{2,}\b"),
        placeholder="[REDACTED:email]",
        severity=SensitivityLevel.MEDIUM,
        description="Email address",
    ),

    # ── PII: High+ ───────────────────────────────────────────────────────────
    # NOTE: credit_card is listed before phone_number even though it requires HIGH
    # sensitivity — a 16-digit card number contains a valid 10-digit phone sequence,
    # so the longer, more-specific pattern must win when both are active.
    RedactionRule(
        name="credit_card",
        pattern=_p(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12})\b"),
        placeholder="[REDACTED:credit_card]",
        severity=SensitivityLevel.HIGH,
        description="Credit card number (Luhn-format)",
    ),
    RedactionRule(
        name="phone_number",
        pattern=_p(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        placeholder="[REDACTED:phone]",
        severity=SensitivityLevel.MEDIUM,
        description="US phone number",
    ),
    RedactionRule(
        name="ssn",
        pattern=_p(r"\b\d{3}-\d{2}-\d{4}\b"),
        placeholder="[REDACTED:ssn]",
        severity=SensitivityLevel.HIGH,
        description="US Social Security Number",
    ),
    RedactionRule(
        name="private_ip_range",
        pattern=_p(r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b"),
        placeholder="[REDACTED:private_ip]",
        severity=SensitivityLevel.HIGH,
        description="RFC-1918 private IP address",
    ),

    # ── Paranoid: any long random-looking string ──────────────────────────────
    RedactionRule(
        name="long_entropy_token",
        pattern=_p(r"\b[A-Za-z0-9+/=_\-]{40,}\b"),
        placeholder="[REDACTED:high_entropy_token]",
        severity=SensitivityLevel.PARANOID,
        description="High-entropy string (possible secret)",
    ),
]

_SEVERITY_ORDER = {
    SensitivityLevel.LOW: 0,
    SensitivityLevel.MEDIUM: 1,
    SensitivityLevel.HIGH: 2,
    SensitivityLevel.PARANOID: 3,
}


class PIIRedactor:
    """
    Middleware that scans and redacts PII/secrets from checkpoint state
    before it is serialized to disk or Redis.

    Critical: runs BEFORE serialization — secrets never hit storage.

    Args:
        sensitivity: Which categories of PII to scan (default HIGH).
        custom_rules: Additional RedactionRule instances to apply.
        dry_run: If True, detect but do not modify state (for auditing).
        hash_redacted: If True, replace with deterministic hash tokens
                       (allows correlation across checkpoints without re-exposing values).
    """

    def __init__(
        self,
        sensitivity: SensitivityLevel = SensitivityLevel.HIGH,
        custom_rules: list[RedactionRule] | None = None,
        dry_run: bool = False,
        hash_redacted: bool = False,
        extra_backend: Any | None = None,
    ) -> None:
        """
        Args:
            sensitivity: Which categories of PII to scan (default HIGH).
            custom_rules: Additional RedactionRule instances to apply.
            dry_run: If True, detect but do not modify state (for auditing).
            hash_redacted: If True, replace with deterministic hash tokens.
            extra_backend: Optional PresidioBackend (or any object with a
                           ``redact_value(text) -> (str, bool)`` method) for
                           NER-based PII detection on top of regex rules.
        """
        self.sensitivity = sensitivity
        self.dry_run = dry_run
        self.hash_redacted = hash_redacted
        self._extra_backend = extra_backend
        self._level = _SEVERITY_ORDER[sensitivity]

        # Select active rules: all rules at or below our sensitivity level
        self._rules: list[RedactionRule] = [
            r for r in _BUILTIN_RULES
            if _SEVERITY_ORDER[r.severity] <= self._level
        ] + (custom_rules or [])

    def redact_state(self, state_dict: dict[str, Any], session_id: str) -> RedactionResult:
        """
        Scan and redact PII from a serialized state dict IN-PLACE.

        Called automatically by Checkpoint.save() when a redactor is attached.
        Returns a RedactionResult summary (never logs the actual secret values).
        """
        result = RedactionResult(session_id=session_id)
        self._walk_and_redact(state_dict, result, field_path="")
        if result.was_redacted and not self.dry_run:
            logger.info(
                "PIIRedactor: %d redactions in session '%s': %s",
                result.total_redactions, session_id, dict(result.by_category),
            )
        return result

    def redact_text(self, text: str, field_name: str = "") -> tuple[str, list[str]]:
        """
        Redact a single text string. Returns (redacted_text, list_of_categories_found).
        Safe to call on any string — returns original if nothing found.

        If ``extra_backend`` is set (e.g. PresidioBackend), it runs after the
        built-in regex rules to catch contextual PII that regex cannot match.
        """
        found_categories: list[str] = []
        result_text = text

        for rule in self._rules:
            matches = rule.pattern.findall(result_text)
            if matches:
                found_categories.append(rule.name)
                if not self.dry_run:
                    placeholder = self._make_placeholder(rule, matches[0] if isinstance(matches[0], str) else "")
                    result_text = rule.pattern.sub(placeholder, result_text)

        # NER-based backend (optional — Presidio or any redact_value-compatible object)
        if self._extra_backend is not None and not self.dry_run:
            try:
                redacted_by_backend, changed = self._extra_backend.redact_value(result_text)
                if changed:
                    result_text = redacted_by_backend
                    found_categories.append("presidio_ner")
            except Exception as exc:
                logger.warning("extra_backend.redact_value failed: %s", exc)

        return result_text, found_categories

    def scan_only(self, text: str) -> list[str]:
        """Return category names of all detected PII without modifying text."""
        found: list[str] = []
        for rule in self._rules:
            if rule.pattern.search(text):
                found.append(rule.name)
        return found

    # ── Private helpers ──────────────────────────────────────────────────────

    def _make_placeholder(self, rule: RedactionRule, matched_value: str) -> str:
        if self.hash_redacted and matched_value:
            short_hash = hashlib.sha256(matched_value.encode()).hexdigest()[:8]
            return f"{rule.placeholder[:-1]}:{short_hash}]"
        return rule.placeholder

    def _walk_and_redact(
        self, obj: Any, result: RedactionResult, field_path: str
    ) -> Any:
        """Recursively walk a dict/list/str and redact in-place."""
        if isinstance(obj, str):
            redacted, categories = self.redact_text(obj, field_path)
            for cat in categories:
                result.add(cat, field_path, obj, redacted)
            return redacted

        if isinstance(obj, dict):
            for key in list(obj.keys()):
                child_path = f"{field_path}.{key}" if field_path else key
                obj[key] = self._walk_and_redact(obj[key], result, child_path)
            return obj

        if isinstance(obj, list):
            for i, item in enumerate(obj):
                obj[i] = self._walk_and_redact(item, result, f"{field_path}[{i}]")
            return obj

        return obj  # int, float, bool, None — pass through

    def add_rule(self, rule: RedactionRule) -> None:
        """Register an additional custom rule at runtime."""
        self._rules.append(rule)

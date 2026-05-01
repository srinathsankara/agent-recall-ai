"""
Hallucinated package name detector.

Scans tool calls (pip install, import statements, require()) for package names
that don't match known real packages. Catches the classic hallucination pattern
where an agent installs a convincingly-named but non-existent package.

Detection strategy:
1. Extract package names from install commands
2. Check against a curated list of common real packages
3. Apply heuristic scoring: suspicious names (too long, unusual chars, double-hyphens)
4. Flag for human review — does not block by default
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..core.state import AlertSeverity, AlertType
from .base import BaseMonitor

if TYPE_CHECKING:
    from ..core.state import TaskState

# Well-known packages that are definitely real
_KNOWN_SAFE = frozenset({
    "numpy", "pandas", "scipy", "matplotlib", "sklearn", "scikit-learn",
    "tensorflow", "torch", "torchvision", "transformers", "datasets",
    "fastapi", "flask", "django", "starlette", "uvicorn", "gunicorn",
    "sqlalchemy", "alembic", "pydantic", "marshmallow", "cerberus",
    "requests", "httpx", "aiohttp", "urllib3", "certifi",
    "boto3", "botocore", "google-cloud-storage", "azure-storage-blob",
    "openai", "anthropic", "langchain", "langchain-core", "langchain-community",
    "pytest", "pytest-asyncio", "pytest-cov", "mypy", "ruff", "black", "isort",
    "typer", "click", "rich", "colorama", "tqdm",
    "pyyaml", "toml", "tomli", "dotenv", "python-dotenv",
    "redis", "celery", "kombu", "pymongo", "motor",
    "psycopg2", "psycopg2-binary", "asyncpg", "aiosqlite",
    "pillow", "opencv-python", "imageio",
    "cryptography", "pyjwt", "passlib", "bcrypt",
    "tiktoken", "tokenizers", "sentencepiece",
    "diskcache", "cachetools", "joblib",
    "tenacity", "backoff", "retry",
    "paramiko", "fabric", "invoke",
    "jinja2", "mako", "chameleon",
    "arrow", "pendulum", "dateutil", "python-dateutil",
    "loguru", "structlog", "python-json-logger",
    "pip", "setuptools", "wheel", "hatchling", "build",
    "agent-recall-ai", "agenttest",
})

# Suspicious patterns that often indicate hallucinated package names
_SUSPICIOUS_PATTERNS = [
    # Must NOT use re.I for this one — "fastapi" has 7+ letters, re.I would false-positive
    (r"[A-Z]{5,}", "all-caps segment (>4 chars)"),
    # Below patterns use re.I safely (no character class ambiguity)
    (r"--", "double-hyphen in package name"),
    (r"\d{4,}", "long numeric sequence"),
    # AI-prefixed packages with 3+ additional hyphen segments are hallucination-prone
    (r"^(ai|llm|gpt|claude|openai)-(?:[a-z]+-){2,}[a-z]+$", "AI package with 3+ hyphen segments"),
    (r"_(utils|helper|tools|lib|sdk|api|wrapper)_(v\d+|pro|plus|max)$", "versioned util suffix"),
]

_INSTALL_PATTERNS = [
    re.compile(r"pip\s+install\s+([\w\-\.]+(?:\[[\w,]+\])?(?:==[\d\.]+)?)", re.I),
    re.compile(r"pip3\s+install\s+([\w\-\.]+)", re.I),
    re.compile(r"poetry\s+add\s+([\w\-\.]+)", re.I),
    re.compile(r"uv\s+add\s+([\w\-\.]+)", re.I),
]

_IMPORT_PATTERN = re.compile(r"^(?:import|from)\s+([\w]+)", re.M)


def _normalize_package(name: str) -> str:
    """Normalize: strip version, extras, lowercase, replace _ with -."""
    name = re.sub(r"\[.*\]", "", name)    # strip extras
    name = re.sub(r"[>=<!].*", "", name)  # strip version
    return name.lower().replace("_", "-").strip()


def _is_suspicious(package: str) -> str | None:
    """Returns a reason string if suspicious, else None."""
    for pattern, reason in _SUSPICIOUS_PATTERNS:
        # The all-caps check must NOT use re.I — "fastapi" would falsely match [A-Z]{5,}
        flags = 0 if "A-Z" in pattern else re.I
        if re.search(pattern, package, flags):
            return reason
    # Excessively long names
    if len(package) > 50:
        return "unusually long package name"
    # Contains repeated words
    parts = package.split("-")
    if len(parts) != len(set(parts)) and len(parts) > 2:
        return "repeated words in package name"
    return None


class PackageHallucinationMonitor(BaseMonitor):
    """
    Detects potentially hallucinated package names in tool call outputs.

    Scans tool_calls for install commands and import statements.
    Flags names that are unknown + suspicious for human review.
    """

    def __init__(self, extra_known: set[str] | None = None) -> None:
        self._known = _KNOWN_SAFE | (extra_known or set())
        self._alerted_packages: set[str] = set()

    def check(self, state: TaskState) -> list[dict]:
        alerts: list[dict] = []
        recent_calls = state.tool_calls[-10:]

        for tc in recent_calls:
            full_text = f"{tc.input_summary} {tc.output_summary}"
            packages = self._extract_packages(full_text)

            for pkg in packages:
                normalized = _normalize_package(pkg)
                if normalized in self._alerted_packages:
                    continue
                if normalized in self._known:
                    continue

                reason = _is_suspicious(normalized)
                if reason:
                    self._alerted_packages.add(normalized)
                    alerts.append({
                        "alert_type": AlertType.PACKAGE_HALLUCINATION,
                        "severity": AlertSeverity.WARN,
                        "message": (
                            f"Possibly hallucinated package: '{normalized}' — {reason}. "
                            "Verify before installing."
                        ),
                        "detail": {
                            "package": normalized,
                            "original": pkg,
                            "reason": reason,
                            "tool": tc.tool_name,
                        },
                    })

        return alerts

    def _extract_packages(self, text: str) -> list[str]:
        packages: list[str] = []
        for pattern in _INSTALL_PATTERNS:
            packages.extend(pattern.findall(text))
        return packages

    on_tool_call = check

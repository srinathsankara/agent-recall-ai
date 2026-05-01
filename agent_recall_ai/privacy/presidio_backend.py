"""
PresidioBackend — NER-based PII detection via Microsoft Presidio.

An optional upgrade over the regex-only PIIRedactor.  Presidio uses spaCy
Named Entity Recognition plus rule-based recognizers for contextual accuracy.

Install:
    pip install 'agent-recall-ai[presidio]'
    python -m spacy download en_core_web_lg   # recommended
    # or: python -m spacy download en_core_web_sm  (smaller, less accurate)

Why Presidio?
-------------
Regex catches syntactically structured secrets (API keys, SSNs, CCs) but
misses contextual PII like:

    "Contact John Smith at his office in New York"
    "The patient's name is Alice Johnson, DOB 1985-03-12"

Presidio uses NER to catch PERSON, LOCATION, DATE, ORG entities that regex
will never match.

Usage:
    from agent_recall_ai.privacy import PIIRedactor
    from agent_recall_ai.privacy.presidio_backend import PresidioBackend

    backend = PresidioBackend(
        entities=["PERSON", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER"],
        language="en",
    )
    redactor = PIIRedactor(extra_backend=backend)

    with Checkpoint("my-task", redactor=redactor) as cp:
        ...

Supported entity types (subset of Presidio's full list):
    PERSON, LOCATION, ORGANIZATION, DATE_TIME,
    EMAIL_ADDRESS, PHONE_NUMBER, US_SSN, CREDIT_CARD,
    IBAN_CODE, IP_ADDRESS, URL, NRP (Nationality/Religion/Political)
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from presidio_analyzer import AnalyzerEngine, RecognizerResult  # noqa: F401
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    _PRESIDIO_AVAILABLE = True
except ImportError:
    _PRESIDIO_AVAILABLE = False


class PresidioBackend:
    """
    Presidio-powered PII detection and anonymization.

    This is not a standalone redactor — it is a *backend* that is called by
    PIIRedactor when ``extra_backend=PresidioBackend(...)`` is set.

    Parameters
    ----------
    entities:
        List of Presidio entity types to detect.  Defaults to a safe
        production set that covers the most common PII categories.
    language:
        Language code for NER model (default: "en").
    score_threshold:
        Minimum confidence score for a detection to be acted upon (0.0–1.0).
    anonymize_operator:
        Anonymization operator: "replace" (default), "redact", "hash", "mask".
    """

    _DEFAULT_ENTITIES = [
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "US_SSN",
        "CREDIT_CARD",
        "IP_ADDRESS",
        "LOCATION",
        "DATE_TIME",
        "URL",
        "IBAN_CODE",
    ]

    def __init__(
        self,
        entities: list[str] | None = None,
        language: str = "en",
        score_threshold: float = 0.5,
        anonymize_operator: str = "replace",
    ) -> None:
        if not _PRESIDIO_AVAILABLE:
            raise ImportError(
                "Microsoft Presidio is required:\n"
                "    pip install 'agent-recall-ai[presidio]'\n"
                "    python -m spacy download en_core_web_lg"
            )
        self._entities = entities or self._DEFAULT_ENTITIES
        self._language = language
        self._threshold = score_threshold
        self._operator = anonymize_operator

        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        logger.debug("PresidioBackend initialized with entities: %s", self._entities)

    def scan(self, text: str) -> list[dict]:
        """
        Analyze text and return a list of detected PII items.

        Each item: {"entity_type": str, "start": int, "end": int,
                    "score": float, "text": str}
        """
        if not text or not isinstance(text, str):
            return []
        results = self._analyzer.analyze(
            text=text,
            entities=self._entities,
            language=self._language,
            score_threshold=self._threshold,
        )
        return [
            {
                "entity_type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": r.score,
                "text": text[r.start:r.end],
            }
            for r in results
        ]

    def anonymize(self, text: str) -> tuple[str, list[dict]]:
        """
        Detect and anonymize PII in text.

        Returns (anonymized_text, detections).
        """
        if not text or not isinstance(text, str):
            return text, []

        analyzer_results = self._analyzer.analyze(
            text=text,
            entities=self._entities,
            language=self._language,
            score_threshold=self._threshold,
        )
        if not analyzer_results:
            return text, []

        operators = {
            entity: OperatorConfig(self._operator, {"new_value": f"<{entity}>"})
            for entity in self._entities
        }

        anonymized = self._anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,
            operators=operators,
        )
        detections = [
            {
                "entity_type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": r.score,
            }
            for r in analyzer_results
        ]
        return anonymized.text, detections

    def redact_value(self, value: str) -> tuple[str, bool]:
        """
        Redact a single string value.

        Returns (redacted_value, was_changed).
        """
        anonymized, detections = self.anonymize(value)
        if detections:
            logger.debug(
                "Presidio detected %d PII item(s): %s",
                len(detections),
                [d["entity_type"] for d in detections],
            )
        return anonymized, bool(detections)

"""Firewall module — public API for threat detection and PII-safe sanitization."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from .redactor import Redactor, RedactionResult
from .sentinel import Sentinel, SentinelResult

logger = logging.getLogger(__name__)

__all__ = [
    "FirewallSDK",
    "Interceptor",
    "Redactor",
    "RedactionResult",
    "Sentinel",
    "SentinelResult",
    "create_firewall_sdk",
]


class Interceptor:
    def __init__(self, sentinel: Sentinel, redactor: Redactor, weilchain: Any | None = None) -> None:
        self.sentinel = sentinel
        self.redactor = redactor
        self.weilchain = weilchain
        is_test_mode = os.getenv("TEST_MODE", "").lower() == "true" or bool(os.getenv("PYTEST_CURRENT_TEST"))
        self._async_commit = (os.getenv("WEILCHAIN_ASYNC_COMMIT", "true").lower() == "true") and (not is_test_mode)

    def _commit_event(self, **kwargs: Any) -> None:
        if self.weilchain is None or not hasattr(self.weilchain, "commit"):
            return
        is_test_mode = os.getenv("TEST_MODE", "").lower() == "true" or bool(os.getenv("PYTEST_CURRENT_TEST"))
        if (not self._async_commit) or is_test_mode:
            self.weilchain.commit(**kwargs)
            return

        def _runner() -> None:
            try:
                self.weilchain.commit(**kwargs)
            except Exception:
                logger.exception("Async Weilchain commit failed")

        threading.Thread(target=_runner, name="weilchain-commit", daemon=True).start()

    def ingress(self, prompt: str, session_id: str) -> dict:
        trace_id = str(uuid4())
        scan = self.sentinel.scan(prompt)
        if scan["is_threat"]:
            self._commit_event(
                session_id=session_id,
                event_type="INGRESS_BLOCK",
                threat_type=scan["threat_type"],
                layer_used=scan.get("layer_used", "HEURISTIC"),
                confidence=scan["confidence"],
                trace_id=trace_id,
            )
            return {
                "trace_id": trace_id,
                "verdict": "BLOCKED",
                "sanitized_prompt": "",
                "threat_type": scan["threat_type"],
                "confidence": scan["confidence"],
            }

        prompt_redacted = self.redactor.redact(prompt)
        return {
            "trace_id": trace_id,
            "verdict": "CLEAN",
            "sanitized_prompt": prompt_redacted["redacted_text"],
            "threat_type": scan["threat_type"],
            "confidence": scan["confidence"],
        }

    def egress(self, trace_id: str, session_id: str, payload: str) -> dict:
        result = self.redactor.redact(payload)
        verdict = "SUSPICIOUS" if result["redactions"] else "CLEAN"
        if verdict == "SUSPICIOUS":
            self._commit_event(
                session_id=session_id,
                event_type="EGRESS_REDACT",
                threat_type="EGRESS_PII",
                layer_used="NER+REGEX",
                confidence=1.0,
                encrypted_fields=result.get("encrypted_fields", []),
                redacted_fields=[r for r in result["redactions"]
                                 if r not in result.get("encrypted_fields", [])],
                trace_id=trace_id,
            )
        return {
            "trace_id": trace_id,
            "verdict": verdict,
            "sanitized_payload": result["redacted_text"],
            "redactions": result["redactions"],
            "encrypted_fields": result.get("encrypted_fields", []),
        }


@dataclass
class FirewallSDK:
    sentinel: Sentinel
    redactor: Redactor
    interceptor: Interceptor

    def inspect_prompt(self, prompt: str, session_id: str = "default") -> dict[str, Any]:
        return self.interceptor.ingress(prompt, session_id)

    def sanitize_text(self, text: str) -> dict[str, Any]:
        return self.redactor.redact(text)

    def sanitize_response(self, trace_id: str, response_text: str, session_id: str = "default") -> dict[str, Any]:
        return self.interceptor.egress(trace_id, session_id, response_text)


def create_firewall_sdk(load_models: bool = True) -> FirewallSDK:
    sentinel = Sentinel()
    redactor = Redactor()
    if load_models:
        sentinel.load()
        redactor.load()
    interceptor = Interceptor(sentinel=sentinel, redactor=redactor, weilchain=None)
    return FirewallSDK(sentinel=sentinel, redactor=redactor, interceptor=interceptor)

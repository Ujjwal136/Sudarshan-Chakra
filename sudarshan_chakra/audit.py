from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditChain:
    """Lightweight local blockchain for security audit events."""

    def __init__(self, storage_path: str | None = None) -> None:
        default_path = os.getenv("AEGIS_AUDIT_CHAIN_PATH", "aegis_audit_chain.json")
        self._storage_path = Path(storage_path or default_path)
        self._lock = threading.RLock()
        self._chain: list[dict] = []
        self._load()

    def _hash_block(self, index: int, prev_hash: str, event: dict) -> str:
        payload = {
            "index": index,
            "prev_hash": prev_hash,
            "event": event,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _persist(self) -> None:
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"chain": self._chain}
        self._storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self._storage_path.exists():
            return
        try:
            payload = json.loads(self._storage_path.read_text(encoding="utf-8"))
            chain = payload.get("chain", [])
            if isinstance(chain, list):
                self._chain = chain
        except Exception:
            self._chain = []

    def start_background_receipt_polling(self, interval_seconds: float = 2.0) -> None:
        # Kept for API compatibility with the previous audit backend.
        _ = interval_seconds

    def stop_background_receipt_polling(self) -> None:
        # Kept for API compatibility with the previous audit backend.
        return

    def commit(
        self,
        *,
        session_id: str,
        event_type: str,
        threat_type: str,
        layer_used: str = "",
        confidence: float = 0.0,
        encrypted_fields: list[str] | None = None,
        redacted_fields: list[str] | None = None,
        trace_id: str | None = None,
    ) -> None:
        with self._lock:
            index = len(self._chain)
            previous_hash = self._chain[-1]["block_hash"] if self._chain else "0" * 64
            timestamp = _utc_now()
            trace = trace_id or str(uuid4())
            event = {
                "trace_id": trace,
                "session_id": session_id,
                "event_type": event_type,
                "threat_type": threat_type,
                "timestamp_utc": timestamp,
                "encrypted_fields": list(encrypted_fields or []),
                "redacted_fields": list(redacted_fields or []),
                "layer_used": layer_used,
                "confidence": float(confidence),
            }
            block_hash = self._hash_block(index=index, prev_hash=previous_hash, event=event)
            block = {
                "index": index,
                "previous_hash": previous_hash,
                "block_hash": block_hash,
                "event": event,
            }
            self._chain.append(block)
            self._persist()

    def get_all(self) -> list[dict]:
        with self._lock:
            entries: list[dict] = []
            for block in self._chain:
                event = dict(block["event"])
                event["weilchain_hash"] = block["block_hash"]
                event["chain_hash"] = block["block_hash"]
                event["block_height"] = str(block["index"])
                event["batch_id"] = ""
                event["tx_idx"] = str(block["index"])
                event["tx_hash"] = block["block_hash"]
                event["receipt_status"] = "committed"
                event["onchain"] = True
                entries.append(event)
            return entries

    def connectivity(self) -> dict:
        return {
            "status": "online",
            "backend": "aegis_local_chain",
            "storage_path": str(self._storage_path),
        }

    def stats(self) -> dict:
        entries = self.get_all()
        return {
            "total": len(entries),
            "blocked": sum(1 for e in entries if e.get("event_type") in {"BLOCK", "INGRESS_BLOCK"}),
            "redacted": sum(1 for e in entries if e.get("event_type") == "EGRESS_REDACT"),
        }

    def _verify_chain(self) -> tuple[bool, str]:
        with self._lock:
            for i, block in enumerate(self._chain):
                expected_prev = "0" * 64 if i == 0 else self._chain[i - 1]["block_hash"]
                if block.get("previous_hash") != expected_prev:
                    return False, f"previous hash mismatch at index {i}"
                expected_hash = self._hash_block(index=i, prev_hash=expected_prev, event=block.get("event", {}))
                if block.get("block_hash") != expected_hash:
                    return False, f"block hash mismatch at index {i}"
            return True, "ok"

    def verify(self, entry_or_trace_id) -> dict | bool:
        # Backward compatibility: callers may pass a ledger entry dict and expect bool.
        if isinstance(entry_or_trace_id, dict):
            trace_id = str(entry_or_trace_id.get("trace_id", "")).strip()
            current = next((e for e in self.get_all() if e.get("trace_id") == trace_id), None)
            if current is None:
                return False
            candidate = dict(entry_or_trace_id)
            keep_keys = {
                "trace_id",
                "session_id",
                "event_type",
                "threat_type",
                "timestamp_utc",
                "encrypted_fields",
                "redacted_fields",
                "layer_used",
                "confidence",
            }
            current_cmp = {k: current.get(k) for k in keep_keys}
            cand_cmp = {k: candidate.get(k) for k in keep_keys}
            return current_cmp == cand_cmp

        trace_id = str(entry_or_trace_id).strip()
        entry = next((e for e in self.get_all() if e.get("trace_id") == trace_id), None)
        if entry is None:
            return {"error": "trace_id not found", "trace_id": trace_id}
        chain_ok, reason = self._verify_chain()
        return {
            "trace_id": trace_id,
            "valid": bool(chain_ok),
            "reason": reason,
            "entry": entry,
        }

    def verify_all(self) -> dict:
        chain_ok, reason = self._verify_chain()
        return {
            "valid": bool(chain_ok),
            "reason": reason,
            "checked": len(self._chain),
        }


def create_audit_chain(storage_path: str | None = None) -> AuditChain:
    return AuditChain(storage_path=storage_path)

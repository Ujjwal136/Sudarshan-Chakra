"""Tests for AuditChain (local blockchain audit log)."""

import json
from pathlib import Path

from sudarshan_chakra.audit import AuditChain, create_audit_chain


class TestAuditChainCommit:
    def test_commit_adds_block(self, tmp_path):
        chain = AuditChain(storage_path=str(tmp_path / "chain.json"))
        chain.commit(
            session_id="s1",
            event_type="INGRESS_BLOCK",
            threat_type="prompt_injection",
            confidence=0.95,
        )
        entries = chain.get_all()
        assert len(entries) == 1
        assert entries[0]["event_type"] == "INGRESS_BLOCK"
        assert entries[0]["threat_type"] == "prompt_injection"
        assert entries[0]["confidence"] == 0.95

    def test_commit_multiple_blocks(self, tmp_path):
        chain = AuditChain(storage_path=str(tmp_path / "chain.json"))
        for i in range(5):
            chain.commit(
                session_id=f"s{i}",
                event_type="EGRESS_REDACT",
                threat_type="EGRESS_PII",
                confidence=1.0,
            )
        entries = chain.get_all()
        assert len(entries) == 5


class TestAuditChainVerify:
    def test_verify_valid_entry(self, tmp_path):
        chain = AuditChain(storage_path=str(tmp_path / "chain.json"))
        chain.commit(
            session_id="s1",
            event_type="INGRESS_BLOCK",
            threat_type="prompt_injection",
            confidence=0.9,
            trace_id="trace-001",
        )
        result = chain.verify("trace-001")
        assert isinstance(result, dict)
        assert result["valid"] is True
        assert result["trace_id"] == "trace-001"

    def test_verify_missing_trace(self, tmp_path):
        chain = AuditChain(storage_path=str(tmp_path / "chain.json"))
        result = chain.verify("nonexistent")
        assert "error" in result

    def test_verify_dict_entry(self, tmp_path):
        chain = AuditChain(storage_path=str(tmp_path / "chain.json"))
        chain.commit(
            session_id="s1",
            event_type="INGRESS_BLOCK",
            threat_type="prompt_injection",
            confidence=0.9,
            trace_id="trace-002",
        )
        entry = chain.get_all()[0]
        assert chain.verify(entry) is True

    def test_verify_all_intact_chain(self, tmp_path):
        chain = AuditChain(storage_path=str(tmp_path / "chain.json"))
        for i in range(3):
            chain.commit(
                session_id=f"s{i}",
                event_type="EGRESS_REDACT",
                threat_type="EGRESS_PII",
                confidence=1.0,
            )
        result = chain.verify_all()
        assert result["valid"] is True
        assert result["checked"] == 3


class TestAuditChainPersistence:
    def test_persist_and_reload(self, tmp_path):
        path = str(tmp_path / "chain.json")
        chain1 = AuditChain(storage_path=path)
        chain1.commit(
            session_id="s1",
            event_type="INGRESS_BLOCK",
            threat_type="sql_injection",
            confidence=0.8,
            trace_id="trace-persist",
        )
        # Reload from disk
        chain2 = AuditChain(storage_path=path)
        entries = chain2.get_all()
        assert len(entries) == 1
        assert entries[0]["trace_id"] == "trace-persist"

    def test_stats(self, tmp_path):
        chain = AuditChain(storage_path=str(tmp_path / "chain.json"))
        chain.commit(session_id="s1", event_type="INGRESS_BLOCK", threat_type="pi", confidence=0.9)
        chain.commit(session_id="s2", event_type="EGRESS_REDACT", threat_type="PII", confidence=1.0)
        stats = chain.stats()
        assert stats["total"] == 2
        assert stats["blocked"] == 1
        assert stats["redacted"] == 1


class TestCreateAuditChain:
    def test_factory_function(self, tmp_path):
        chain = create_audit_chain(storage_path=str(tmp_path / "factory.json"))
        assert isinstance(chain, AuditChain)
        conn = chain.connectivity()
        assert conn["status"] == "online"

"""Shared test fixtures."""

import os
import pytest


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch, tmp_path):
    """Set environment variables for test isolation."""
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("FPE_KEY", "a" * 64)
    monkeypatch.setenv("FPE_TWEAK", "b" * 14)
    monkeypatch.setenv("AEGIS_AUDIT_CHAIN_PATH", str(tmp_path / "test_audit_chain.json"))

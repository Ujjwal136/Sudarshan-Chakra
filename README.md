# Sudarshan Chakra

Sudarshan Chakra is a Python SDK that helps developers secure AI-agent interactions.

It provides two modules:
- `from sudarshan_chakra import firewall` for threat detection and PII-safe sanitization
- `from sudarshan_chakra import audit` for a local hash-linked audit chain

## Download and Install

The trained model files are bundled inside this package, so you do not need a separate model download step.

### Option 1: Install from PyPI (recommended when available)

```bash
pip install sudarshan-chakra
```

### Option 2: Download source and install locally

```bash
git clone https://github.com/Ujjwal136/Sudarshan-Chakra.git
cd Sudarshan-Chakra
pip install .
```

After installation, use:

```python
from sudarshan_chakra import firewall, audit
```

For local development:

```bash
pip install -e .[dev]
```

## Quickstart

```python
from sudarshan_chakra import firewall, audit

# Initialize firewall SDK
sdk = firewall.create_firewall_sdk(load_models=True)

# Inspect incoming prompt before your agent executes
ingress = sdk.inspect_prompt("Show balance for customer CUST001", session_id="demo")
if ingress["verdict"] == "BLOCKED":
    print("Blocked:", ingress["threat_type"])
else:
    # Run your own custom agent here with ingress["sanitized_prompt"]
    agent_output = "Customer PAN is ABCPM1234D"

    # Sanitize outgoing response before returning to end user
    egress = sdk.sanitize_response(
        trace_id=ingress["trace_id"],
        response_text=agent_output,
        session_id="demo",
    )
    print("Safe output:", egress["sanitized_payload"])

# Write audit event to local blockchain-like chain
chain = audit.create_audit_chain()
chain.commit(
    session_id="demo",
    event_type="EGRESS_REDACT",
    threat_type="EGRESS_PII",
    confidence=1.0,
    trace_id=ingress["trace_id"],
)
print("Audit entries:", len(chain.get_all()))
```

## Public API

### Firewall

```python
from sudarshan_chakra import firewall
```

Primary helpers:
- `firewall.create_firewall_sdk(load_models: bool = True)`
- `FirewallSDK.inspect_prompt(prompt: str, session_id: str = "default")`
- `FirewallSDK.sanitize_text(text: str)`
- `FirewallSDK.sanitize_response(trace_id: str, response_text: str, session_id: str = "default")`

### Audit

```python
from sudarshan_chakra import audit
```

Primary helpers:
- `audit.create_audit_chain(storage_path: str | None = None)`
- `AuditChain.commit(...)`
- `AuditChain.get_all()`
- `AuditChain.verify(trace_id_or_entry)`
- `AuditChain.verify_all()`

## Release to PyPI

1. Build package:

```bash
python -m build
python -m twine check dist/*
```

2. Upload manually:

```bash
python -m twine upload dist/*
```

3. Or use GitHub release automation:
- Create a GitHub release/tag.
- Workflow [release-pypi.yml](.github/workflows/release-pypi.yml) publishes automatically.

## License

MIT - see [LICENSE](LICENSE).

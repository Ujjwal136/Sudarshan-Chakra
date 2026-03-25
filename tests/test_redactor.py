"""Tests for Redactor regex PII detection patterns."""

from sudarshan_chakra.redactor import Redactor


class TestAadhaarRedaction:
    def setup_method(self):
        self.redactor = Redactor()

    def test_aadhaar_with_spaces(self):
        result = self.redactor.redact("My Aadhaar is 1234 5678 9012")
        assert "1234 5678 9012" not in result["redacted_text"]
        assert "AADHAAR" in result["redactions"]

    def test_aadhaar_without_spaces(self):
        result = self.redactor.redact("Aadhaar: 123456789012")
        assert "123456789012" not in result["redacted_text"]


class TestPANRedaction:
    def setup_method(self):
        self.redactor = Redactor()

    def test_pan_standard(self):
        result = self.redactor.redact("PAN card: ABCPM1234D")
        assert "ABCPM1234D" not in result["redacted_text"]
        assert "PAN" in result["redactions"]


class TestPhoneRedaction:
    def setup_method(self):
        self.redactor = Redactor()

    def test_phone_10digit(self):
        result = self.redactor.redact("Call me at 9876543210")
        assert "9876543210" not in result["redacted_text"]
        assert "PHONE" in result["redactions"]

    def test_phone_with_country_code(self):
        result = self.redactor.redact("Phone: +91 9876543210")
        assert "9876543210" not in result["redacted_text"]


class TestEmailRedaction:
    def setup_method(self):
        self.redactor = Redactor()

    def test_email_detected(self):
        result = self.redactor.redact("Email: user@example.com")
        assert "user@example.com" not in result["redacted_text"]
        assert "EMAIL" in result["redactions"]


class TestIFSCRedaction:
    def setup_method(self):
        self.redactor = Redactor()

    def test_ifsc_code(self):
        result = self.redactor.redact("IFSC: SBIN0001234")
        assert "SBIN0001234" not in result["redacted_text"]
        assert "IFSC" in result["redactions"]


class TestDOBRedaction:
    def setup_method(self):
        self.redactor = Redactor()

    def test_dob_dd_mm_yyyy(self):
        result = self.redactor.redact("DOB: 15/08/1990")
        assert "15/08/1990" not in result["redacted_text"]
        assert "DOB" in result["redactions"]

    def test_dob_iso_format(self):
        result = self.redactor.redact("Birth date: 1990-08-15")
        assert "1990-08-15" not in result["redacted_text"]


class TestRedactReturnStructure:
    def setup_method(self):
        self.redactor = Redactor()

    def test_clean_text_returns_no_redactions(self):
        result = self.redactor.redact("Hello, this is a normal message.")
        assert result["redacted_text"] == "Hello, this is a normal message."
        assert result["redactions"] == []

    def test_multiple_pii_types(self):
        text = "User ABCPM1234D called from 9876543210 with email a@b.com"
        result = self.redactor.redact(text)
        assert "ABCPM1234D" not in result["redacted_text"]
        assert "9876543210" not in result["redacted_text"]
        assert len(result["redactions"]) >= 2

    def test_result_has_required_keys(self):
        result = self.redactor.redact("test")
        assert "redacted_text" in result
        assert "redactions" in result
        assert "encrypted_fields" in result

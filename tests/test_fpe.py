"""Tests for FPE (Format-Preserving Encryption) engine."""

from sudarshan_chakra.fpe_engine import (
    encrypt_aadhaar, decrypt_aadhaar,
    encrypt_pan, decrypt_pan,
    encrypt_phone, decrypt_phone,
    encrypt_ifsc, decrypt_ifsc,
    encrypt_account_no, decrypt_account_no,
    FPEEngine,
)


class TestAadhaarFPE:
    def test_roundtrip(self):
        original = "123456789012"
        encrypted = encrypt_aadhaar(original)
        assert not encrypted.startswith("[")  # not a fallback
        decrypted = decrypt_aadhaar(encrypted)
        # Remove spaces for comparison
        assert decrypted.replace(" ", "") == original

    def test_roundtrip_with_spaces(self):
        original = "1234 5678 9012"
        encrypted = encrypt_aadhaar(original)
        assert not encrypted.startswith("[")
        decrypted = decrypt_aadhaar(encrypted)
        assert decrypted.replace(" ", "") == "123456789012"

    def test_invalid_aadhaar_falls_back(self):
        result = encrypt_aadhaar("12345")
        assert result == "[AADHAAR_REDACTED]"

    def test_non_numeric_falls_back(self):
        result = encrypt_aadhaar("ABCDEF123456")
        assert result == "[AADHAAR_REDACTED]"


class TestPANFPE:
    def test_roundtrip(self):
        original = "ABCPM1234D"
        encrypted = encrypt_pan(original)
        assert not encrypted.startswith("[")
        assert len(encrypted) == 10
        decrypted = decrypt_pan(encrypted)
        assert decrypted == original

    def test_invalid_pan_falls_back(self):
        result = encrypt_pan("SHORT")
        assert result == "[PAN_REDACTED]"


class TestPhoneFPE:
    def test_roundtrip_10digit(self):
        original = "9876543210"
        encrypted = encrypt_phone(original)
        assert not encrypted.startswith("[")
        decrypted = decrypt_phone(encrypted)
        assert decrypted.replace("+91 ", "").strip() == original

    def test_roundtrip_with_country_code(self):
        original = "+91 9876543210"
        encrypted = encrypt_phone(original)
        assert not encrypted.startswith("[")

    def test_invalid_phone_falls_back(self):
        result = encrypt_phone("123")
        assert result == "[PHONE_REDACTED]"


class TestIFSCFPE:
    def test_roundtrip(self):
        original = "SBIN0001234"
        encrypted = encrypt_ifsc(original)
        assert not encrypted.startswith("[")
        assert len(encrypted) == 11
        decrypted = decrypt_ifsc(encrypted)
        assert decrypted == original

    def test_invalid_ifsc_falls_back(self):
        result = encrypt_ifsc("SHORT")
        assert result == "[IFSC_REDACTED]"


class TestAccountNoFPE:
    def test_roundtrip(self):
        original = "12345678901"  # 11 digits
        encrypted = encrypt_account_no(original)
        assert not encrypted.startswith("[")
        decrypted = decrypt_account_no(encrypted)
        assert decrypted == original

    def test_too_short_falls_back(self):
        result = encrypt_account_no("1234567890")  # 10 digits
        assert result == "[ACCOUNT_NO_REDACTED]"


class TestFPEEngineWrapper:
    def test_encrypt_decrypt_aadhaar_without_spaces(self):
        engine = FPEEngine()
        original = "123456789012"
        encrypted = engine.encrypt(original, "AADHAAR")
        assert encrypted != original
        decrypted = engine.decrypt(encrypted, "AADHAAR")
        assert decrypted == original

    def test_unknown_entity_passthrough(self):
        engine = FPEEngine()
        result = engine.encrypt("hello", "UNKNOWN")
        assert result == "hello"

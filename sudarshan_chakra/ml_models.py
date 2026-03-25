"""
Inference-only ML model classes and feature extraction utilities.

These are extracted from the training scripts so the published package
does not ship training-time dependencies or code.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np

# ── Sentinel Layer A: SGD Classifier ────────────────────────────────────────

TOKEN_RE = re.compile(r"\b\w+\b", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def _extract_raw_features(text: str) -> list[str]:
    """Extract unigrams, bigrams, and character trigrams."""
    tokens = tokenize(text)
    features: list[str] = [f"w:{t}" for t in tokens]
    for i in range(len(tokens) - 1):
        features.append(f"bi:{tokens[i]}_{tokens[i + 1]}")
    for tok in tokens:
        for j in range(max(0, len(tok) - 2)):
            features.append(f"c3:{tok[j:j+3]}")
    return features


def vectorize(text: str, vocab: dict[str, int]) -> dict[int, float]:
    """Convert text to a sparse feature vector {index: value}."""
    vec: dict[int, float] = {}
    for feat in _extract_raw_features(text):
        idx = vocab.get(feat)
        if idx is not None:
            vec[idx] = vec.get(idx, 0.0) + 1.0
    return vec


class SGDClassifier:
    """Binary logistic regression trained with SGD (inference only)."""

    def __init__(self, n_features: int, lr: float = 0.01, alpha: float = 1e-4) -> None:
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = lr
        self.alpha = alpha

    @staticmethod
    def _sigmoid(z: float) -> float:
        z = max(min(z, 50.0), -50.0)
        return 1.0 / (1.0 + math.exp(-z))

    def _raw_score(self, x: dict[int, float]) -> float:
        s = self.bias
        for idx, val in x.items():
            s += self.weights[idx] * val
        return s

    def predict_proba(self, x: dict[int, float]) -> float:
        return self._sigmoid(self._raw_score(x))

    def predict(self, x: dict[int, float]) -> int:
        return 1 if self.predict_proba(x) >= 0.5 else 0

    def to_dict(self) -> dict:
        return {"weights": self.weights, "bias": self.bias}

    @classmethod
    def from_dict(cls, d: dict) -> SGDClassifier:
        obj = cls(n_features=len(d["weights"]))
        obj.weights = d["weights"]
        obj.bias = d["bias"]
        return obj


# ── Sentinel Layer B: MLP ───────────────────────────────────────────────────

N_META = 9  # must match len(_meta_features)


def _char_ngrams(text: str, ns: tuple[int, ...] = (4, 5)) -> list[str]:
    """Extract character-level n-grams from raw text."""
    lowered = text.lower()
    grams: list[str] = []
    for n in ns:
        for i in range(max(0, len(lowered) - n + 1)):
            grams.append(lowered[i:i + n])
    return grams


def _meta_features(text: str) -> list[float]:
    """Handcrafted features that capture prompt structure."""
    length = len(text)
    words = TOKEN_RE.findall(text)
    n_words = max(len(words), 1)
    upper_chars = sum(1 for c in text if c.isupper())
    digit_chars = sum(1 for c in text if c.isdigit())
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    avg_word_len = sum(len(w) for w in words) / n_words if words else 0

    return [
        min(length / 200.0, 1.0),
        min(n_words / 30.0, 1.0),
        upper_chars / max(length, 1),
        digit_chars / max(length, 1),
        special_chars / max(length, 1),
        avg_word_len / 15.0,
        1.0 if text.isupper() else 0.0,
        1.0 if "?" in text else 0.0,
        1.0 if "!" in text else 0.0,
    ]


def vectorize_b(text: str, vocab: dict[str, int], idf: np.ndarray) -> np.ndarray:
    """Convert text to TF-IDF char n-gram vector + meta features."""
    tf = np.zeros(len(vocab), dtype=np.float32)
    for gram in _char_ngrams(text):
        idx = vocab.get(gram)
        if idx is not None:
            tf[idx] += 1.0
    tf = np.log1p(tf) * idf
    norm = np.linalg.norm(tf)
    if norm > 0:
        tf /= norm
    meta = np.array(_meta_features(text), dtype=np.float32)
    return np.concatenate([tf, meta])


class MLP:
    """
    Input(n_features) → Dense(128, ReLU) → Dense(1, Sigmoid)
    Inference-only wrapper.
    """

    def __init__(self, n_input: int, n_hidden: int = 128) -> None:
        self.n_input = n_input
        self.n_hidden = n_hidden
        scale1 = math.sqrt(2.0 / n_input)
        scale2 = math.sqrt(2.0 / n_hidden)
        self.W1 = (np.random.randn(n_input, n_hidden) * scale1).astype(np.float32)
        self.b1 = np.zeros(n_hidden, dtype=np.float32)
        self.W2 = (np.random.randn(n_hidden, 1) * scale2).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass. X shape: (batch, n_input). Returns (output, hidden, pre_act)."""
        z1 = X @ self.W1 + self.b1
        h1 = np.maximum(z1, 0)
        z2 = h1 @ self.W2 + self.b2
        out = 1.0 / (1.0 + np.exp(-np.clip(z2, -50, 50)))
        return out, h1, z1

    def predict_proba(self, x: np.ndarray) -> float:
        """Single sample → probability of attack."""
        X = x.reshape(1, -1)
        out, _, _ = self.forward(X)
        return float(out[0, 0])

    def predict(self, x: np.ndarray) -> int:
        return 1 if self.predict_proba(x) >= 0.5 else 0

    def to_dict(self) -> dict:
        return {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "n_input": self.n_input,
            "n_hidden": self.n_hidden,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MLP:
        obj = cls(d["n_input"], d["n_hidden"])
        obj.W1 = np.array(d["W1"], dtype=np.float32)
        obj.b1 = np.array(d["b1"], dtype=np.float32)
        obj.W2 = np.array(d["W2"], dtype=np.float32)
        obj.b2 = np.array(d["b2"], dtype=np.float32)
        return obj


# ── Redactor NER: Feature extraction ────────────────────────────────────────

def word_shape(word: str) -> str:
    """Map word to shape: 'Xxxxx', 'dddd', 'x.x@x', etc."""
    s = []
    for ch in word[:20]:
        if ch.isupper():
            s.append("X")
        elif ch.islower():
            s.append("x")
        elif ch.isdigit():
            s.append("d")
        else:
            s.append(ch)
    return "".join(s)


def extract_features(tokens: list[str], idx: int, prev_label: str) -> dict[str, float]:
    """Extract features for the token at position idx."""
    word = tokens[idx]
    features: dict[str, float] = {
        "bias": 1.0,
        f"word.lower={word.lower()}": 1.0,
        f"word.shape={word_shape(word)}": 1.0,
        f"word.isupper={word.isupper()}": 1.0,
        f"word.istitle={word.istitle()}": 1.0,
        f"word.isdigit={word.isdigit()}": 1.0,
        f"word.isalpha={word.isalpha()}": 1.0,
        f"word.len={min(len(word), 15)}": 1.0,
        f"prev_label={prev_label}": 1.0,
    }

    for n in (1, 2, 3):
        if len(word) >= n:
            features[f"prefix{n}={word[:n].lower()}"] = 1.0
            features[f"suffix{n}={word[-n:].lower()}"] = 1.0

    if re.match(r"\d{4}\s?\d{4}\s?\d{4}", word):
        features["pattern=aadhaar_like"] = 1.0
    if re.match(r"[A-Z]{5}\d{4}[A-Z]", word):
        features["pattern=pan_like"] = 1.0
    if re.match(r"\+?\d[\d\-\s]{8,}", word):
        features["pattern=phone_like"] = 1.0
    if "@" in word:
        features["pattern=has_at"] = 1.0
    if re.match(r"\d{4}-\d{2}-\d{2}", word) or re.match(r"\d{2}/\d{2}/\d{4}", word):
        features["pattern=date_like"] = 1.0
    if re.match(r"[A-Z]{4}0[A-Z0-9]{6}", word):
        features["pattern=ifsc_like"] = 1.0
    if re.match(r"[A-Z]\d{7}$", word):
        features["pattern=passport_like"] = 1.0
    if re.match(r"\d{11,16}$", word):
        features["pattern=account_like"] = 1.0

    if idx > 0:
        prev = tokens[idx - 1]
        features[f"prev.lower={prev.lower()}"] = 1.0
        features[f"prev.istitle={prev.istitle()}"] = 1.0
        features[f"prev.shape={word_shape(prev)}"] = 1.0
    else:
        features["BOS"] = 1.0

    if idx < len(tokens) - 1:
        nxt = tokens[idx + 1]
        features[f"next.lower={nxt.lower()}"] = 1.0
        features[f"next.istitle={nxt.istitle()}"] = 1.0
        features[f"next.shape={word_shape(nxt)}"] = 1.0
    else:
        features["EOS"] = 1.0

    if idx > 1:
        features[f"prev2.lower={tokens[idx-2].lower()}"] = 1.0
    if idx < len(tokens) - 2:
        features[f"next2.lower={tokens[idx+2].lower()}"] = 1.0

    return features

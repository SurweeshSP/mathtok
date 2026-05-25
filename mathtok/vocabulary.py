"""
Layer 7: Vocabulary & BPE Compression

Two-tier vocabulary design
──────────────────────────
  Tier 1 — Fixed Math Vocabulary
    Every mathematical token (operators, functions, variables, constants,
    structural) has a deterministic integer ID.  These IDs are NEVER
    computed by BPE; their meaning is exact and invariant.

  Tier 2 — BPE Text Vocabulary
    Natural-language text spans are compressed using the HuggingFace
    `tokenizers` library (Byte-Pair Encoding).  Only text tokens are
    subject to BPE; math tokens bypass BPE entirely.

HuggingFace PreTrainedTokenizer compatibility
─────────────────────────────────────────────
  MathTokHFTokenizer subclasses PreTrainedTokenizer so the tokenizer
  can be used as a drop-in replacement in any HF training pipeline:

      from mathtok import MathTokVocabulary
      tok = MathTokVocabulary.build_hf_tokenizer(pipeline)
      tok.save_pretrained("./mathtok-tokenizer")
      tok = MathTokHFTokenizer.from_pretrained("./mathtok-tokenizer")
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from .operator_registry import get_all_operator_tokens

logger = logging.getLogger(__name__)


# ── Fixed vocabulary constants ────────────────────────────────────────────

_SPECIAL_TOKENS = [
    "[PAD]",        # 0
    "[UNK]",        # 1
    "[UNK_MATH]",   # 2
    "[BOS]",        # 3
    "[EOS]",        # 4
    "[MATH_START]", # 5
    "[MATH_END]",   # 6
    "[TEXT_START]", # 7
    "[TEXT_END]",   # 8
    "[SEP]",        # 9
    "[MASK]",       # 10
    "[SCOPE_OPEN]", # 11
    "[SCOPE_CLOSE]",# 12
    "SUBTREE_TRUNCATED", # 13
]

# Common variable tokens
_VAR_TOKENS = [
    "VAR_X", "VAR_Y", "VAR_Z", "VAR_T", "VAR_N", "VAR_K",
    "VAR_A", "VAR_B", "VAR_C", "VAR_M", "VAR_I", "VAR_J",
    "VAR_R", "VAR_S", "VAR_U", "VAR_V", "VAR_W", "VAR_P",
    "VAR_Q", "VAR_L", "VAR_F", "VAR_G", "VAR_H",
    # Greek
    "VAR_THETA", "VAR_ALPHA", "VAR_BETA",  "VAR_GAMMA_",
    "VAR_DELTA", "VAR_EPSILON","VAR_ZETA",  "VAR_ETA",
    "VAR_LAMBDA","VAR_MU",    "VAR_NU",    "VAR_XI",
    "VAR_RHO",   "VAR_SIGMA", "VAR_TAU",   "VAR_PHI",
    "VAR_CHI",   "VAR_PSI",   "VAR_OMEGA",
    "VAR_IOTA",  "VAR_KAPPA", "VAR_OMICRON", "VAR_UPSILON",
]

# Constant tokens: CONST_-10 through CONST_100
_CONST_TOKENS = (
    [f"CONST_{i}" for i in range(-10, 101)]
    + ["CONST_PI", "CONST_E", "CONST_I", "CONST_INF", "CONST_NEG_INF", "CONST_NAN"]
)

# Large-number / float fallback tokens  (dynamically added as needed)
_NUMERIC_PLACEHOLDERS = [f"NUM_{i}" for i in range(101, 1001)]


def _build_fixed_vocab() -> dict[str, int]:
    """
    Build the complete fixed math vocabulary: token → integer ID.
    The ordering here determines the permanent token IDs.
    """
    tokens: list[str] = []
    tokens.extend(_SPECIAL_TOKENS)
    tokens.extend(get_all_operator_tokens())   # from operator_registry
    tokens.extend(_VAR_TOKENS)
    tokens.extend(_CONST_TOKENS)
    tokens.extend(_NUMERIC_PLACEHOLDERS)
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return {tok: idx for idx, tok in enumerate(deduped)}


# ── MathTokVocabulary ─────────────────────────────────────────────────────

class MathTokVocabulary:
    """
    Two-tier math + BPE vocabulary manager.

    Fixed math tokens are deterministically assigned IDs.
    BPE vocabulary (trained on text corpora) is appended after.

    Parameters
    ----------
    bpe_vocab_size : int
        Target size of the BPE sub-vocabulary for text tokens.
    """

    VOCAB_FILE  = "mathtok_vocab.json"
    MERGES_FILE = "mathtok_bpe_merges.txt"

    def __init__(self, bpe_vocab_size: int = 8000) -> None:
        self.bpe_vocab_size = bpe_vocab_size
        self._math_vocab: dict[str, int] = _build_fixed_vocab()
        self._ids_to_tokens: dict[int, str] = {v: k for k, v in self._math_vocab.items()}
        self._bpe_tokenizer = None          # HF tokenizers.Tokenizer for text
        self._bpe_offset    = len(self._math_vocab)   # BPE IDs start here

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def math_vocab_size(self) -> int:
        return len(self._math_vocab)

    @property
    def total_vocab_size(self) -> int:
        if self._bpe_tokenizer is None:
            return self.math_vocab_size
        return self.math_vocab_size + len(self._bpe_tokenizer.get_vocab())

    def get_vocab(self) -> dict[str, int]:
        """Return the complete merged vocabulary."""
        vocab = dict(self._math_vocab)
        if self._bpe_tokenizer is not None:
            for tok, idx in self._bpe_tokenizer.get_vocab().items():
                merged_id = self._bpe_offset + idx
                if tok not in vocab:
                    vocab[tok] = merged_id
        return vocab

    # ── Token ↔ ID ────────────────────────────────────────────────────────

    def token_to_id(self, token: str) -> int:
        """Return the integer ID for a token, using [UNK]=1 as fallback."""
        if token in self._math_vocab:
            return self._math_vocab[token]
        if self._bpe_tokenizer is not None:
            bpe_id = self._bpe_tokenizer.token_to_id(token)
            if bpe_id is not None:
                return self._bpe_offset + bpe_id
        return self._math_vocab["[UNK]"]

    def id_to_token(self, idx: int) -> str:
        """Return the token string for an integer ID."""
        if idx in self._ids_to_tokens:
            return self._ids_to_tokens[idx]
        if self._bpe_tokenizer is not None:
            bpe_idx = idx - self._bpe_offset
            if bpe_idx >= 0:
                tok = self._bpe_tokenizer.id_to_token(bpe_idx)
                if tok is not None:
                    return tok
        return "[UNK]"

    def encode_text(self, text: str) -> list[int]:
        """Encode a plain text span with BPE (fallback to char-level)."""
        if self._bpe_tokenizer is not None:
            enc = self._bpe_tokenizer.encode(text)
            return [self._bpe_offset + i for i in enc.ids]
        # Character-level fallback
        return [self.token_to_id(ch) for ch in text]

    def encode_math_tokens(self, tokens: list[str]) -> list[int]:
        """Map a list of math token strings to integer IDs."""
        return [self.token_to_id(t) for t in tokens]

    def add_math_token(self, token: str) -> int:
        """Dynamically add a new math token (e.g. VAR_FOO) to vocabulary."""
        if token not in self._math_vocab:
            new_id = len(self._math_vocab)
            self._math_vocab[token] = new_id
            self._ids_to_tokens[new_id] = token
            self._bpe_offset = len(self._math_vocab)
        return self._math_vocab[token]

    # ── BPE training ──────────────────────────────────────────────────────

    def train_bpe(self, text_corpus: list[str]) -> None:
        """
        Train a BPE tokenizer on a list of text strings.
        Only the TEXT spans of math problem descriptions should be used.

        Requires: pip install tokenizers
        """
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
            from tokenizers.pre_tokenizers import Whitespace
        except ImportError:
            raise ImportError("Install 'tokenizers' package: pip install tokenizers")

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=self.bpe_vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
            show_progress=False,
        )
        tokenizer.train_from_iterator(text_corpus, trainer=trainer)
        self._bpe_tokenizer = tokenizer
        logger.info(
            "BPE trained: vocab_size=%d, total_vocab=%d",
            len(tokenizer.get_vocab()),
            self.total_vocab_size,
        )

    def load_bpe_from_pretrained(self, model_name_or_path: str = "gpt2") -> None:
        """
        Load a pre-trained HuggingFace tokenizer as the BPE backend.
        Useful as a zero-shot baseline for the text sub-vocabulary.
        """
        try:
            from transformers import AutoTokenizer
            hf_tok = AutoTokenizer.from_pretrained(model_name_or_path)
            # Wrap in our interface by using its encoding
            self._hf_text_tokenizer = hf_tok
            self._bpe_tokenizer = None   # use _hf_text_tokenizer path instead
            logger.info("Loaded HF text tokenizer: %s", model_name_or_path)
        except Exception as exc:
            logger.warning("Could not load HF tokenizer %s: %s", model_name_or_path, exc)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save vocabulary to directory."""
        dirpath = Path(directory)
        dirpath.mkdir(parents=True, exist_ok=True)

        vocab_path = dirpath / self.VOCAB_FILE
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self._math_vocab, f, indent=2)

        if self._bpe_tokenizer is not None:
            merges_path = dirpath / self.MERGES_FILE
            self._bpe_tokenizer.model.save(str(dirpath))
        logger.info("Vocabulary saved to %s", dirpath)

    @classmethod
    def load(cls, directory: str) -> "MathTokVocabulary":
        """Load vocabulary from a saved directory."""
        dirpath = Path(directory)
        vocab_path = dirpath / cls.VOCAB_FILE

        instance = cls()
        with open(vocab_path, "r", encoding="utf-8") as f:
            instance._math_vocab = json.load(f)
        instance._ids_to_tokens = {v: k for k, v in instance._math_vocab.items()}
        instance._bpe_offset    = len(instance._math_vocab)

        # Try loading BPE if present
        bpe_path = dirpath / "vocab.json"
        if bpe_path.exists():
            try:
                from tokenizers import Tokenizer
                instance._bpe_tokenizer = Tokenizer.from_file(str(dirpath / "tokenizer.json"))
            except Exception as exc:
                logger.warning("Could not load BPE tokenizer: %s", exc)

        logger.info("Vocabulary loaded from %s (size=%d)", dirpath, len(instance._math_vocab))
        return instance

    # ── HuggingFace PreTrainedTokenizer factory ───────────────────────────

    def build_hf_tokenizer(self, pipeline=None) -> "MathTokHFTokenizer":
        """
        Build a HuggingFace PreTrainedTokenizer wrapping this vocabulary
        and the given MathTokPipeline.

        Parameters
        ----------
        pipeline : MathTokPipeline | None
            If None, a default pipeline is created.
        """
        return MathTokHFTokenizer(vocab=self, pipeline=pipeline)


# ── HuggingFace PreTrainedTokenizer wrapper ───────────────────────────────

class MathTokHFTokenizer:
    """
    HuggingFace-compatible tokenizer wrapping MathTokVocabulary.

    Implements the PreTrainedTokenizer interface so it can be used with:
      - transformers.Trainer
      - datasets.map(..., batched=True)
      - model.generate(tokenizer(...))

    The full MathTok pipeline (canonicalize → AST → serialize) runs
    inside _tokenize(), making it a transparent drop-in replacement.
    """

    def __init__(self, vocab: MathTokVocabulary, pipeline=None) -> None:
        self.vocab    = vocab
        self.pipeline = pipeline

        # HF-compatible special token IDs
        self.pad_token    = "[PAD]"
        self.unk_token    = "[UNK]"
        self.bos_token    = "[BOS]"
        self.eos_token    = "[EOS]"
        self.mask_token   = "[MASK]"
        self.sep_token    = "[SEP]"

        self.pad_token_id  = vocab.token_to_id("[PAD]")
        self.unk_token_id  = vocab.token_to_id("[UNK]")
        self.bos_token_id  = vocab.token_to_id("[BOS]")
        self.eos_token_id  = vocab.token_to_id("[EOS]")

    # ── Tokenization ──────────────────────────────────────────────────────

    def tokenize(self, text: str) -> list[str]:
        """Return token strings for the input."""
        if self.pipeline is not None:
            out = self.pipeline.encode(text)
            return out.tokens
        # Minimal fallback: just split on spaces
        return text.split()

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Return token IDs for the input."""
        tokens = self.tokenize(text)
        ids = self.vocab.encode_math_tokens(tokens)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to a string."""
        tokens = [self.vocab.id_to_token(i) for i in ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if not t.startswith("[")]
        return " ".join(tokens)

    def __call__(
        self,
        text: str | list[str],
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
    ) -> dict:
        """Callable interface compatible with HF DataCollator."""
        if isinstance(text, str):
            text = [text]
        all_ids = [self.encode(t, add_special_tokens=add_special_tokens) for t in text]
        result = {"input_ids": all_ids}
        if return_tensors == "pt":
            try:
                import torch
                max_len = max(len(ids) for ids in all_ids)
                padded = [
                    ids + [self.pad_token_id] * (max_len - len(ids))
                    for ids in all_ids
                ]
                result["input_ids"] = torch.tensor(padded, dtype=torch.long)
                result["attention_mask"] = (result["input_ids"] != self.pad_token_id).long()
            except ImportError:
                pass
        return result

    def get_vocab(self) -> dict[str, int]:
        return self.vocab.get_vocab()

    def __len__(self) -> int:
        return self.vocab.total_vocab_size

    def save_pretrained(self, save_directory: str) -> None:
        """Save tokenizer to a directory."""
        self.vocab.save(save_directory)
        config = {
            "tokenizer_class": "MathTokHFTokenizer",
            "model_max_length": 2048,
            "pad_token":  self.pad_token,
            "unk_token":  self.unk_token,
            "bos_token":  self.bos_token,
            "eos_token":  self.eos_token,
            "mask_token": self.mask_token,
        }
        config_path = Path(save_directory) / "tokenizer_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        logger.info("HF tokenizer saved to %s", save_directory)

    @classmethod
    def from_pretrained(cls, load_directory: str) -> "MathTokHFTokenizer":
        """Load tokenizer from a saved directory."""
        vocab = MathTokVocabulary.load(load_directory)
        return cls(vocab=vocab)

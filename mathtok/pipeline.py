"""
End-to-end MathTok Pipeline

Orchestrates all 7 layers into a single encode() call.

Pipeline flow
─────────────
  Input text
    → HybridLexer             (split TEXT / MATH spans)
    → For each MATH span:
        → Canonicalizer       (normalize expression)
        → ASTGenerator        (SymPy → ASTNode tree)
        → StructuralSerializer (DFS → SerializedToken list)
        → MetadataGenerator   (structural attention metadata)
        → MathTokVocabulary   (token → ID)
    → For each TEXT span:
        → MathTokVocabulary.encode_text() (BPE)
    → Merge results into TokenizedOutput

Usage
─────
  >>> from mathtok import MathTokPipeline
  >>> p = MathTokPipeline()
  >>> out = p.encode("The derivative of $\\sin(x^2) + 3x$")
  >>> out.tokens           # list[str]
  >>> out.input_ids        # list[int]
  >>> out.metadata         # list[TokenMetadata]
  >>> out.sexp             # S-expression string (math spans only)

CLI
───
  python -m mathtok.pipeline "sin(x^2) + 3x"
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from .canonicalizer  import Canonicalizer, CanonicalizationResult
from .lexer          import HybridLexer, SpanType, LexSpan
from .ast_generator  import ASTGenerator, ASTNode
from .serializer     import StructuralSerializer, SerializedToken
from .metadata       import MetadataGenerator, TokenMetadata
from .vocabulary     import MathTokVocabulary

logger = logging.getLogger(__name__)


# ── Output dataclass ──────────────────────────────────────────────────────

@dataclass
class TokenizedOutput:
    """
    Complete output of the MathTok pipeline for one input string.

    Attributes
    ----------
    tokens     : Merged token string sequence (math + text tokens).
    input_ids  : Corresponding vocabulary integer IDs.
    metadata   : Structural metadata for each token position.
    spans      : Original LexSpan objects (TEXT / MATH segments).
    math_sexps : S-expression strings for each MATH span.
    canon_results : CanonicalizationResult per MATH span.
    warnings   : Any non-fatal warnings from the pipeline.
    """
    tokens:        list[str]               = field(default_factory=list)
    input_ids:     list[int]               = field(default_factory=list)
    metadata:      list[TokenMetadata]     = field(default_factory=list)
    spans:         list[LexSpan]           = field(default_factory=list)
    math_sexps:    list[str]               = field(default_factory=list)
    canon_results: list[CanonicalizationResult] = field(default_factory=list)
    warnings:      list[str]              = field(default_factory=list)

    @property
    def sexp(self) -> str:
        """Join all math S-expressions with a space."""
        return "  ".join(self.math_sexps)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Tokens      : {len(self.tokens)}",
            f"Math spans  : {len(self.math_sexps)}",
            f"Vocab IDs   : {self.input_ids[:10]}{'...' if len(self.input_ids) > 10 else ''}",
            f"S-expression: {self.sexp[:120]}",
        ]
        if self.warnings:
            lines.append(f"Warnings    : {'; '.join(self.warnings)}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "tokens":    self.tokens,
            "input_ids": self.input_ids,
            "metadata":  [m.to_dict() for m in self.metadata],
            "math_sexps": self.math_sexps,
            "warnings":  self.warnings,
        }


# ── Main pipeline ─────────────────────────────────────────────────────────

class MathTokPipeline:
    """
    End-to-end tokenization pipeline for mixed text+math input.

    Parameters
    ----------
    canonicalizer : Canonicalizer | None
        Override the default canonicalizer.
    lexer : HybridLexer | None
        Override the default lexer.
    ast_generator : ASTGenerator | None
        Override the default AST generator.
    serializer : StructuralSerializer | None
        Override the default serializer.
    metadata_gen : MetadataGenerator | None
        Override the default metadata generator.
    vocab : MathTokVocabulary | None
        Override the default vocabulary.
    include_metadata : bool
        Whether to compute structural metadata (slightly slower).
    """

    def __init__(
        self,
        canonicalizer:    Optional[Canonicalizer]           = None,
        lexer:            Optional[HybridLexer]             = None,
        ast_generator:    Optional[ASTGenerator]            = None,
        serializer:       Optional[StructuralSerializer]    = None,
        metadata_gen:     Optional[MetadataGenerator]       = None,
        vocab:            Optional[MathTokVocabulary]       = None,
        include_metadata: bool = True,
        timeout_seconds: float = 5.0,
        max_depth: int = 20,
        emit_scope_tokens: bool = True,
    ) -> None:
        self.canon     = canonicalizer  or Canonicalizer(timeout_seconds=timeout_seconds)
        self.lexer     = lexer          or HybridLexer()
        self.ast_gen   = ast_generator  or ASTGenerator(max_depth=max_depth)
        self.serializer= serializer     or StructuralSerializer(emit_scope_tokens=emit_scope_tokens)
        self.meta_gen  = metadata_gen   or MetadataGenerator()
        self.vocab     = vocab          or MathTokVocabulary()
        self.include_metadata = include_metadata

    # ── Public API ────────────────────────────────────────────────────────

    def encode(self, text: str) -> TokenizedOutput:
        """
        Tokenize a mixed text+math string through the full pipeline.

        Parameters
        ----------
        text : str
            Input containing natural language and/or mathematical
            expressions in LaTeX or ASCII format.

        Returns
        -------
        TokenizedOutput
        """
        out = TokenizedOutput()
        spans = self.lexer.lex(text)
        out.spans = spans

        all_serialized: list[SerializedToken] = []

        for span in spans:
            if span.span_type is SpanType.MATH:
                ser_tokens, sexp, canon_result, warnings = self._process_math(span.content)
                out.math_sexps.append(sexp)
                out.canon_results.append(canon_result)
                out.warnings.extend(warnings)
                all_serialized.extend(ser_tokens)
                out.tokens.extend(st.token for st in ser_tokens)
                out.input_ids.extend(self.vocab.token_to_id(st.token) for st in ser_tokens)
            else:
                text_ids = self.vocab.encode_text(span.content.strip())
                text_tokens = [self.vocab.id_to_token(i) for i in text_ids]
                out.tokens.extend(text_tokens)
                out.input_ids.extend(text_ids)

        # Structural metadata
        if self.include_metadata and all_serialized:
            vocab_map = self.vocab.get_vocab()
            out.metadata = self.meta_gen.generate(all_serialized, vocab=vocab_map)

        return out

    def encode_batch(self, texts: list[str]) -> list[TokenizedOutput]:
        """Tokenize a list of strings."""
        return [self.encode(t) for t in texts]

    def encode_math_only(self, expression: str) -> TokenizedOutput:
        """
        Directly tokenize a pure math expression (no lexer splitting).
        Use when the input is guaranteed to be a single math expression.
        """
        ser_tokens, sexp, canon_result, warnings = self._process_math(expression)
        out = TokenizedOutput(
            tokens      = [st.token for st in ser_tokens],
            input_ids   = [self.vocab.token_to_id(st.token) for st in ser_tokens],
            math_sexps  = [sexp],
            canon_results = [canon_result],
            warnings    = warnings,
        )
        if self.include_metadata and ser_tokens:
            vocab_map = self.vocab.get_vocab()
            out.metadata = self.meta_gen.generate(ser_tokens, vocab=vocab_map)
        return out

    def get_hf_tokenizer(self):
        """Return a HuggingFace-compatible tokenizer wrapper."""
        return self.vocab.build_hf_tokenizer(pipeline=self)

    # ── Math processing sub-pipeline ──────────────────────────────────────

    def _process_math(
        self, expression: str
    ) -> tuple[list[SerializedToken], str, CanonicalizationResult, list[str]]:
        """
        Run a single math expression through:
          Canonicalize → AST → Serialize → (metadata later)

        Returns (serialized_tokens, sexp_string, canon_result, warnings)
        """
        warnings: list[str] = []

        # Step 1: Canonicalize
        canon_result = self.canon.canonicalize(expression)
        warnings.extend(canon_result.warnings)

        if not canon_result.success:
            # Emit a single error token so downstream doesn't break
            error_tok = SerializedToken(
                token="[UNK]", position=0, depth=0, node_id=-1,
                parent_id=-1, child_index=0, num_children=0,
                is_leaf=True, subtree_size=1,
            )
            return [error_tok], "[UNK]", canon_result, warnings

        # Step 2: Build AST
        try:
            ast_root = self.ast_gen.generate(canon_result.expr)
        except Exception as exc:
            warnings.append(f"AST generation failed: {exc}")
            error_tok = SerializedToken(
                token="[UNK]", position=0, depth=0, node_id=-1,
                parent_id=-1, child_index=0, num_children=0,
                is_leaf=True, subtree_size=1,
            )
            return [error_tok], "[UNK]", canon_result, warnings

        # Step 3: Serialize to flat token stream
        try:
            ser_tokens = self.serializer.serialize(ast_root)
            sexp       = self.serializer.to_sexp(ast_root)
        except Exception as exc:
            warnings.append(f"Serialization failed: {exc}")
            return [], "", canon_result, warnings

        # Step 4: Dynamically register any new variable tokens
        for st in ser_tokens:
            if st.token.startswith("VAR_") or st.token.startswith("NUM_"):
                self.vocab.add_math_token(st.token)

        return ser_tokens, sexp, canon_result, warnings


# ── CLI ───────────────────────────────────────────────────────────────────

def cli() -> None:
    """Command-line interface for quick testing."""
    parser = argparse.ArgumentParser(
        description="MathTok: Tokenize a mathematical expression."
    )
    parser.add_argument("expression", nargs="?", help="Math expression to tokenize")
    parser.add_argument("--json",  action="store_true", help="Output full JSON")
    parser.add_argument("--sexp",  action="store_true", help="Output S-expression only")
    args = parser.parse_args()

    text = args.expression or input("Expression: ")

    pipeline = MathTokPipeline()
    out      = pipeline.encode(text)

    if args.json:
        print(json.dumps(out.to_dict(), indent=2))
    elif args.sexp:
        print(out.sexp)
    else:
        print(out.summary())
        print("\nTokens:", out.tokens)


if __name__ == "__main__":
    cli()

"""
MathTok: A Hybrid Canonicalized AST-Based Tokenization Framework
for Mathematical Language Modeling.

Paper: "MathTok: A Hybrid Canonicalized AST-Based Tokenization Framework
        for Mathematical Language Modeling"

Pipeline stages
───────────────
  1. Canonicalization  — normalize mathematically equivalent forms
  2. Hybrid Lexer      — split text / math spans (LaTeX + ASCII)
  3. AST Generator     — SymPy expression → typed ASTNode tree
  4. Operator Registry — semantic metadata per operator/function
  5. Serializer        — DFS preorder flattening of tree
  6. Metadata          — per-token structural attention hints
  7. Vocabulary        — fixed math vocab + BPE text; HF-compatible
"""

from .pipeline         import MathTokPipeline
from .canonicalizer    import Canonicalizer, CanonicalizationResult
from .lexer            import HybridLexer, LexSpan, SpanType
from .ast_generator    import ASTGenerator, ASTNode
from .operator_registry import OPERATOR_REGISTRY, OperatorMeta, get_operator, get_all_operator_tokens, INVERSE_PAIRS
from .serializer       import StructuralSerializer, SerializedToken
from .metadata         import MetadataGenerator, TokenMetadata
from .vocabulary       import MathTokVocabulary, MathTokHFTokenizer
from .validator        import RoundTripValidator, ValidationResult
from .streaming        import MathTokStreamingPipeline

__version__ = "0.1.0"
__all__ = [
    "MathTokPipeline",
    "Canonicalizer", "CanonicalizationResult",
    "HybridLexer", "LexSpan", "SpanType",
    "ASTGenerator", "ASTNode",
    "OperatorMeta", "OPERATOR_REGISTRY", "get_operator", "get_all_operator_tokens", "INVERSE_PAIRS",
    "StructuralSerializer", "SerializedToken",
    "MetadataGenerator", "TokenMetadata",
    "MathTokVocabulary", "MathTokHFTokenizer",
    "RoundTripValidator", "ValidationResult",
    "MathTokStreamingPipeline",
]

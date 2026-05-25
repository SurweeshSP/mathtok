import logging
from typing import Iterator, Optional, Iterable

from .pipeline import TokenizedOutput, MathTokPipeline
from .canonicalizer import Canonicalizer
from .lexer import HybridLexer
from .ast_generator import ASTGenerator
from .serializer import StructuralSerializer
from .metadata import MetadataGenerator
from .vocabulary import MathTokVocabulary

logger = logging.getLogger(__name__)


class MathTokStreamingPipeline:
    """
    A memory-efficient streaming wrapper for MathTokPipeline.
    Uses generators to process massive datasets (e.g., millions of equations)
    without loading all inputs or outputs into RAM simultaneously.
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
        self.pipeline = MathTokPipeline(
            canonicalizer=canonicalizer,
            lexer=lexer,
            ast_generator=ast_generator,
            serializer=serializer,
            metadata_gen=metadata_gen,
            vocab=vocab,
            include_metadata=include_metadata,
            timeout_seconds=timeout_seconds,
            max_depth=max_depth,
            emit_scope_tokens=emit_scope_tokens,
        )

    def encode_stream(self, text_stream: Iterable[str]) -> Iterator[TokenizedOutput]:
        """
        Lazily tokenize a stream of text strings.

        Yields TokenizedOutput instances one at a time.
        """
        for text in text_stream:
            try:
                yield self.pipeline.encode(text)
            except Exception as e:
                logger.warning(f"Failed to encode text {text[:50]!r}: {e}")
                # Yield an empty output or skip? We'll yield an empty one with warning.
                yield TokenizedOutput(warnings=[str(e)])

    def encode_file(self, file_path: str, encoding: str = 'utf-8') -> Iterator[TokenizedOutput]:
        """
        Stream expressions from a line-delimited text file.
        """
        def line_generator() -> Iterator[str]:
            with open(file_path, 'r', encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
                        
        return self.encode_stream(line_generator())

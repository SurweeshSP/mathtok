"""
Layer 2: Hybrid Mathematical Lexer

Splits mixed text+math input into alternating typed spans:
  - TEXT spans  → forwarded to the BPE text tokenizer
  - MATH spans  → forwarded to the canonicalization + AST pipeline

Detection strategy (two-stage)
───────────────────────────────
  Stage 1 — LaTeX delimiter detection
    $...$   $$...$$   \\(...\\)   \\[...\\]
    These are unambiguous; inner content is always MATH.

  Stage 2 — ASCII math heuristic detection
    Applied only to remaining TEXT spans.
    Looks for patterns like:  sin(x),  x^2,  a+b=c,  3*x+1

Outputs a flat ordered list of LexSpan objects.
Adjacent spans of the same type are merged before returning.

Example
───────
  >>> lex = HybridLexer()
  >>> lex.lex("The derivative of $\\\\sin(x^2)$ plus 3x")
  [TEXT("The derivative of "), MATH("\\sin(x^2)"), TEXT(" plus "), MATH("3x")]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterator


# ── Types ──────────────────────────────────────────────────────────────────

class SpanType(str, Enum):
    TEXT = "TEXT"
    MATH = "MATH"


@dataclass
class LexSpan:
    """A contiguous span of homogeneous content type."""
    span_type: SpanType
    content:   str
    start:     int          # character offset in original string
    end:       int
    confidence: float = 1.0 # 0.0 to 1.0

    def __repr__(self) -> str:
        preview = self.content[:50].replace("\n", " ")
        return f"{self.span_type.value}({preview!r}, conf={self.confidence:.2f})"

    def __len__(self) -> int:
        return len(self.content)


# ── Compiled regex patterns ────────────────────────────────────────────────

# Stage 1 — LaTeX delimiters (ordered: longer/greedier patterns first)
_PAT_DISPLAY_DOLLAR   = re.compile(r"\$\$(.+?)\$\$",   re.DOTALL)
_PAT_INLINE_DOLLAR    = re.compile(r"\$(.+?)\$",         re.DOTALL)
_PAT_DISPLAY_BRACKET  = re.compile(r"\\\[(.+?)\\\]",    re.DOTALL)
_PAT_INLINE_PAREN     = re.compile(r"\\\((.+?)\\\)",    re.DOTALL)

_LATEX_PATTERNS = [
    _PAT_DISPLAY_DOLLAR,    # must come before inline dollar
    _PAT_INLINE_DOLLAR,
    _PAT_DISPLAY_BRACKET,
    _PAT_INLINE_PAREN,
]

# Stage 2 — ASCII math heuristic sub-patterns
# Matches: function calls,  exponentiation,  arithmetic expressions
_ASCII_FUNC_CALL = re.compile(
    r"\b(?:sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|"
    r"exp|log|ln|sqrt|cbrt|abs|floor|ceil|"
    r"lim|sum|prod|int|diff|derivative|integral|limit|"
    r"gamma|factorial)\s*\(",
    re.IGNORECASE,
)
_ASCII_EXPONENT = re.compile(
    r"[a-zA-Z_]\w*\s*(?:\^|\*\*)\s*[\w(]"
)
_ASCII_ARITH = re.compile(
    r"(?<!\w)[-+]?\d+(?:\.\d+)?\s*[+\-*/]\s*[-+]?\d"
)
_ASCII_EQUATION = re.compile(
    r"[a-zA-Z_]\w*\s*[+\-*/^=<>]\s*[a-zA-Z0-9_]"
)
_ASCII_FUNCTION_DEF = re.compile(
    r"\b[a-zA-Z_]\w*\([a-zA-Z0-9_,\s]*\)\s*="
)
_ASCII_GREEK = re.compile(
    r"\b(?:alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)\b",
    re.IGNORECASE
)

_ASCII_PATTERNS = [
    _ASCII_FUNC_CALL, _ASCII_EXPONENT, _ASCII_ARITH, _ASCII_EQUATION,
    _ASCII_FUNCTION_DEF, _ASCII_GREEK
]

# Characters that can appear in an ASCII math expression context
_MATH_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                  "0123456789+-*/^=<>()[]{}.,_! \t")


# ── Main class ────────────────────────────────────────────────────────────

class HybridLexer:
    """
    Split mixed text+math input into LexSpan objects.

    Parameters
    ----------
    ascii_math_detection : bool
        Enable Stage-2 heuristic detection inside TEXT spans.
    min_math_len : int
        Minimum character length for an ASCII math span to be emitted
        as MATH (prevents false positives on short strings like "a+b").
    """

    def __init__(
        self,
        ascii_math_detection: bool = True,
        min_math_len: int = 3,
    ) -> None:
        self.ascii_math_detection = ascii_math_detection
        self.min_math_len = min_math_len

    # ── Public API ────────────────────────────────────────────────────────

    def lex(self, text: str) -> list[LexSpan]:
        """
        Lex a mixed text+math string into typed spans.

        Parameters
        ----------
        text : str
            Input string containing natural language and/or math.

        Returns
        -------
        list[LexSpan]
            Ordered list of TEXT and MATH spans.
        """
        if not text:
            return []

        spans = self._stage1_latex(text)

        if self.ascii_math_detection:
            refined: list[LexSpan] = []
            for span in spans:
                if span.span_type is SpanType.TEXT:
                    refined.extend(self._stage2_ascii(span))
                else:
                    refined.append(span)
            spans = refined

        return _merge_adjacent(spans)

    def iter_spans(self, text: str) -> Iterator[LexSpan]:
        """Lazy iterator over lexed spans."""
        yield from self.lex(text)

    def is_math_only(self, text: str) -> bool:
        """Return True if the entire string is a math expression."""
        spans = self.lex(text)
        return all(s.span_type is SpanType.MATH for s in spans if s.content.strip())

    # ── Stage 1: LaTeX delimiter detection ───────────────────────────────

    def _stage1_latex(self, text: str) -> list[LexSpan]:
        """Find all LaTeX-delimited math regions, fill gaps with TEXT."""
        matches: list[tuple[int, int, str]] = []   # (start, end, inner_content)

        for pat in _LATEX_PATTERNS:
            for m in pat.finditer(text):
                s, e = m.start(), m.end()
                # Skip if overlapping with already found match
                if any(not (e <= ms or s >= me) for ms, me, _ in matches):
                    continue
                matches.append((s, e, m.group(1)))   # group(1) = inner content

        matches.sort(key=lambda t: t[0])

        spans: list[LexSpan] = []
        cursor = 0
        for start, end, content in matches:
            if start > cursor:
                spans.append(LexSpan(SpanType.TEXT, text[cursor:start], cursor, start, confidence=1.0))
            spans.append(LexSpan(SpanType.MATH, content.strip(), start, end, confidence=1.0))
            cursor = end

        if cursor < len(text):
            spans.append(LexSpan(SpanType.TEXT, text[cursor:], cursor, len(text), confidence=1.0))

        return spans or [LexSpan(SpanType.TEXT, text, 0, len(text), confidence=1.0)]

    # ── Stage 2: ASCII math detection ────────────────────────────────────

    def _stage2_ascii(self, text_span: LexSpan) -> list[LexSpan]:
        """Within a TEXT span, identify and extract ASCII math regions."""
        text = text_span.content
        base = text_span.start

        math_ranges: list[tuple[int, int]] = []
        for pat in _ASCII_PATTERNS:
            for m in pat.finditer(text):
                s, e = m.start(), m.end()
                s, e = self._expand_region(text, s, e)
                math_ranges.append((s, e))

        if not math_ranges:
            return [text_span]

        math_ranges = _merge_ranges(math_ranges)

        spans: list[LexSpan] = []
        cursor = 0
        for s, e in math_ranges:
            if s > cursor:
                spans.append(LexSpan(SpanType.TEXT, text[cursor:s], base + cursor, base + s, confidence=1.0))
            content = text[s:e].strip()
            
            # Simple heuristic confidence based on length
            # Short strings are less likely to be purely math (e.g., variable names vs full equations)
            conf = min(0.95, max(0.5, 0.5 + 0.05 * len(content)))
            
            span_type = SpanType.MATH if len(content) >= self.min_math_len else SpanType.TEXT
            spans.append(LexSpan(span_type, text[s:e], base + s, base + e, confidence=conf if span_type == SpanType.MATH else 1.0))
            cursor = e

        if cursor < len(text):
            spans.append(LexSpan(SpanType.TEXT, text[cursor:], base + cursor, base + len(text), confidence=1.0))

        return spans

    def _expand_region(self, text: str, start: int, end: int) -> tuple[int, int]:
        """
        Expand a detected math seed region to capture surrounding balanced
        parentheses and chained operators.
        """
        # Expand backwards: include leading unary minus, digits, spaces
        while start > 0 and text[start - 1] in "(-+0123456789 \t":
            if text[start - 1] == "(":
                break
            start -= 1

        # Expand forwards: follow balanced parens and math characters
        depth = 0
        i = end
        while i < len(text):
            ch = text[i]
            if ch in "([{":
                depth += 1
                i += 1
            elif ch in ")]}":
                if depth == 0:
                    break
                depth -= 1
                i += 1
            elif ch in " \t" and depth == 0:
                # Stop at word boundary outside parens
                # — but keep going if next char is still math-ish
                if i + 1 < len(text) and text[i + 1] in "+-*/^=<>)":
                    i += 1
                else:
                    break
            elif ch in _MATH_CHARS:
                i += 1
            else:
                break

        return start, i


# ── Module helpers ────────────────────────────────────────────────────────

def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping (start, end) integer ranges."""
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [list(ranges[0])]
    for s, e in ranges[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(r) for r in merged]


def _merge_adjacent(spans: list[LexSpan]) -> list[LexSpan]:
    """Merge adjacent spans of the same type."""
    if not spans:
        return []
    merged = [spans[0]]
    for span in spans[1:]:
        prev = merged[-1]
        if span.span_type is prev.span_type:
            merged[-1] = LexSpan(
                prev.span_type,
                prev.content + span.content,
                prev.start,
                span.end,
                confidence=max(prev.confidence, span.confidence)
            )
        else:
            merged.append(span)
    return merged

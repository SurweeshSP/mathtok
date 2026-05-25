"""
Tests for the Hybrid Lexer (Layer 2).
"""

import pytest
from mathtok.lexer import HybridLexer, LexSpan, SpanType


@pytest.fixture
def lex():
    return HybridLexer(ascii_math_detection=True, min_math_len=3)


class TestLatexDetection:
    def test_inline_dollar(self, lex):
        spans = lex.lex("Let $x^2 + 1$ be given.")
        types = [s.span_type for s in spans if s.content.strip()]
        assert SpanType.MATH in types
        assert SpanType.TEXT in types

    def test_display_dollar(self, lex):
        spans = lex.lex("$$x^2 + y^2 = 1$$")
        math_spans = [s for s in spans if s.span_type is SpanType.MATH]
        assert len(math_spans) >= 1
        assert "x^2" in math_spans[0].content or "x" in math_spans[0].content

    def test_inline_paren(self, lex):
        spans = lex.lex("We have \\(a + b\\) here.")
        math_spans = [s for s in spans if s.span_type is SpanType.MATH]
        assert len(math_spans) == 1

    def test_display_bracket(self, lex):
        spans = lex.lex("Result: \\[x = \\frac{-b}{2a}\\]")
        math_spans = [s for s in spans if s.span_type is SpanType.MATH]
        assert len(math_spans) == 1

    def test_multiple_math_spans(self, lex):
        spans = lex.lex("If $a > 0$ and $b < 0$, then $a + b$ may be zero.")
        math_spans = [s for s in spans if s.span_type is SpanType.MATH]
        assert len(math_spans) == 3

    def test_pure_text(self, lex):
        spans = lex.lex("This is plain English text with no math at all.")
        math_spans = [s for s in spans if s.span_type is SpanType.MATH]
        assert len(math_spans) == 0


class TestAsciiDetection:
    def test_function_call(self, lex):
        spans = lex.lex("Compute sin(x) for x = pi.")
        math_spans = [s for s in spans if s.span_type is SpanType.MATH]
        assert any("sin" in s.content for s in math_spans)

    def test_exponentiation(self, lex):
        spans = lex.lex("The value of x^2 is always positive.")
        math_spans = [s for s in spans if s.span_type is SpanType.MATH]
        assert len(math_spans) >= 1

    def test_equation(self, lex):
        spans = lex.lex("Solve x^2 + 2*x + 1 = 0.")
        math_spans = [s for s in spans if s.span_type is SpanType.MATH]
        assert len(math_spans) >= 1


class TestEdgeCases:
    def test_empty_string(self, lex):
        spans = lex.lex("")
        assert spans == []

    def test_only_whitespace(self, lex):
        spans = lex.lex("   ")
        assert all(s.span_type is SpanType.TEXT for s in spans)

    def test_is_math_only_true(self, lex):
        assert lex.is_math_only("$x^2 + 1$")

    def test_adjacent_spans_merged(self, lex):
        spans = lex.lex("hello world, no math here at all.")
        # All-text should be merged into a minimal number of spans
        text_spans = [s for s in spans if s.span_type is SpanType.TEXT]
        assert len(text_spans) <= 2

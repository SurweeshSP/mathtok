"""
Integration tests for the end-to-end MathTok Pipeline.
"""

import pytest
from mathtok.pipeline import MathTokPipeline, TokenizedOutput


@pytest.fixture(scope="module")
def pipeline():
    return MathTokPipeline(include_metadata=True)


class TestBasicEncode:
    def test_returns_output(self, pipeline):
        out = pipeline.encode("x^2 + 1")
        assert isinstance(out, TokenizedOutput)

    def test_tokens_nonempty(self, pipeline):
        out = pipeline.encode("sin(x)")
        assert len(out.tokens) > 0

    def test_input_ids_match_tokens(self, pipeline):
        out = pipeline.encode("x^2 + 2*x + 1")
        assert len(out.tokens) == len(out.input_ids)

    def test_ids_are_integers(self, pipeline):
        out = pipeline.encode("x + 1")
        assert all(isinstance(i, int) for i in out.input_ids)

    def test_no_negative_ids(self, pipeline):
        out = pipeline.encode("x + 1")
        # All IDs should be non-negative (UNK=1 is minimum valid)
        assert all(i >= 0 for i in out.input_ids)


class TestMathSpans:
    def test_math_start_end_tokens(self, pipeline):
        out = pipeline.encode("x^2")
        assert "[MATH_START]" in out.tokens
        assert "[MATH_END]" in out.tokens

    def test_sexp_nonempty(self, pipeline):
        out = pipeline.encode("x^2 + 1")
        assert len(out.sexp) > 0

    def test_sexp_contains_op(self, pipeline):
        out = pipeline.encode("x^2")
        assert "OP_POW" in out.sexp

    def test_canon_results(self, pipeline):
        # Use a simple ASCII expression guaranteed to parse successfully
        out = pipeline.encode("x^2 + 1")
        assert len(out.canon_results) >= 1
        assert out.canon_results[0].success


class TestMixedInput:
    def test_mixed_latex(self, pipeline):
        out = pipeline.encode("The result is $x^2 + 1$.")
        assert len(out.tokens) > 0

    def test_mixed_ascii(self, pipeline):
        out = pipeline.encode("Compute sin(x) for x = pi.")
        assert len(out.tokens) > 0

    def test_multiple_math_spans(self, pipeline):
        out = pipeline.encode("If $a > 0$ and $b < 0$ then $a + b$ can be zero.")
        # Should have at least some math tokens
        math_toks = [t for t in out.tokens if t.startswith("OP_") or t.startswith("VAR_")]
        assert len(math_toks) > 0


class TestMetadata:
    def test_metadata_present(self, pipeline):
        out = pipeline.encode("x + 1")
        assert len(out.metadata) > 0

    def test_metadata_positions_sequential(self, pipeline):
        out = pipeline.encode("x^2 + 1")
        positions = [m.position for m in out.metadata]
        assert positions == sorted(positions)

    def test_metadata_categories(self, pipeline):
        out = pipeline.encode("x + 1")
        categories = {m.token_category for m in out.metadata}
        assert "operator" in categories or "variable" in categories or "constant" in categories

    def test_tree_position_keys(self, pipeline):
        out = pipeline.encode("x + 1")
        keys = [m.tree_position_key for m in out.metadata if m.node_id >= 0]
        assert len(keys) > 0
        assert all(isinstance(k, str) for k in keys)


class TestEncodeMathOnly:
    def test_encode_math_only(self, pipeline):
        out = pipeline.encode_math_only("x^2 + 2*x + 1")
        assert len(out.tokens) > 0
        assert "OP_ADD" in out.tokens or "OP_POW" in out.tokens

    def test_encode_batch(self, pipeline):
        exprs = ["x + 1", "sin(x)", "x^2"]
        outs = pipeline.encode_batch(exprs)
        assert len(outs) == 3
        assert all(len(o.tokens) > 0 for o in outs)


class TestHFTokenizer:
    def test_hf_tokenizer_callable(self, pipeline):
        hf_tok = pipeline.get_hf_tokenizer()
        result = hf_tok("x^2 + 1")
        assert "input_ids" in result
        assert len(result["input_ids"]) == 1

    def test_hf_tokenizer_encode(self, pipeline):
        hf_tok = pipeline.get_hf_tokenizer()
        ids = hf_tok.encode("sin(x)")
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_hf_vocab_size(self, pipeline):
        hf_tok = pipeline.get_hf_tokenizer()
        assert len(hf_tok) > 100

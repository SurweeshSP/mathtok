"""
Tests for the Semantic Tokenizer Comparison Framework.
"""

import pytest
from evaluation.comparison import (
    TokenizerStats, ComparisonRecord, TokenizerComparison,
    _score_char, _score_gpt2, _score_mathtok,
    _jaccard, _mean,
    STANDARD_EXPRESSIONS, DEEP_NESTING_EXPRESSIONS, CANONICAL_PAIRS,
)
from mathtok.pipeline import MathTokPipeline


@pytest.fixture(scope="module")
def pipeline():
    return MathTokPipeline(include_metadata=True)


@pytest.fixture(scope="module")
def comp(pipeline):
    return TokenizerComparison(pipeline, gpt2_fn=None, save_jsonl=False)


# ── TokenizerStats ────────────────────────────────────────────────────────

class TestTokenizerStats:
    def test_scr_computed(self):
        stats = TokenizerStats(
            name="test", tokens=["OP_ADD", "VAR_X", "CONST_1"],
            token_count=3,
            operator_nodes=1, tree_depth=1,
            parent_child_relations=1, function_scope=0,
            canonical_bonus=2,
        )
        stats.compute_scr()
        assert stats.structural_score == 5          # 1+1+1+0+2
        assert abs(stats.raw_scr - 5/3) < 1e-9
        assert abs(stats.structural_efficiency - 1/3) < 1e-9

    def test_zero_token_count_safe(self):
        stats = TokenizerStats(name="empty", tokens=[], token_count=0)
        stats.compute_scr()
        assert stats.raw_scr == 0.0


# ── Character-level scorer ─────────────────────────────────────────────────

class TestCharScore:
    def test_simple(self):
        stats = _score_char("x + 1")
        assert stats.token_count == 5
        assert stats.operator_nodes >= 1    # at least +
        assert stats.raw_scr >= 0

    def test_nested_parens_depth(self):
        stats = _score_char("sin((x+1)^2)")
        assert stats.tree_depth >= 2        # at least 2 levels of parens

    def test_no_function_scope(self):
        # Character-level can't identify functions
        stats = _score_char("sin(x)")
        assert stats.function_scope == 0


# ── GPT-2 heuristic scorer ─────────────────────────────────────────────────

class TestGPT2Score:
    def test_operators_detected(self):
        tokens = ["(", "x", "+", "1", ")", "^", "2"]
        stats = _score_gpt2(tokens)
        assert stats.operator_nodes >= 1

    def test_function_detected(self):
        tokens = ["sin", "(", "x", ")"]
        stats = _score_gpt2(tokens)
        assert stats.function_scope >= 1

    def test_paren_depth(self):
        tokens = ["(", "(", "x", ")", ")"]
        stats = _score_gpt2(tokens)
        assert stats.tree_depth == 2

    def test_scr_positive(self):
        tokens = ["sin", "(", "x", "^", "2", ")"]
        stats = _score_gpt2(tokens)
        stats.compute_scr()
        assert stats.raw_scr >= 0


# ── MathTok scorer ────────────────────────────────────────────────────────

class TestMathTokScore:
    def test_add_expression(self, pipeline):
        out = pipeline.encode_math_only("x + 1")
        stats = _score_mathtok(out)
        assert stats.token_count > 0
        assert stats.operator_nodes >= 1    # OP_ADD
        assert stats.canonical_bonus == 2   # successful parse

    def test_function_expression(self, pipeline):
        out = pipeline.encode_math_only("sin(x^2)")
        stats = _score_mathtok(out)
        assert stats.function_scope >= 1    # FUNC_SIN

    def test_depth_nonzero(self, pipeline):
        out = pipeline.encode_math_only("sin(x^2 + 1)")
        stats = _score_mathtok(out)
        assert stats.tree_depth >= 2

    def test_scr_computed(self, pipeline):
        out = pipeline.encode_math_only("(x+1)^2")
        stats = _score_mathtok(out)
        assert stats.raw_scr > 0

    def test_mathtok_scr_higher_than_char(self, pipeline):
        expr = "sin(x^2 + 1)"
        out = pipeline.encode_math_only(expr)
        mt  = _score_mathtok(out)
        ch  = _score_char(expr)
        # MathTok should have higher SCR due to semantic richness
        assert mt.raw_scr > ch.raw_scr


# ── Comparison mechanics ──────────────────────────────────────────────────

class TestComparison:
    def test_compare_one(self, comp):
        rec = comp._compare_one("x + 1", "test")
        assert isinstance(rec, ComparisonRecord)
        assert rec.mathtok.token_count > 0
        assert rec.char_level.token_count > 0
        assert rec.gpt2 is None              # no GPT-2 in fixture

    def test_scr_improvement_vs_char(self, comp):
        rec = comp._compare_one("sin(x^2)", "test")
        # MathTok should outperform char-level on SCR
        assert rec.scr_improvement_vs_char > 0

    def test_canonical_jaccard(self, comp, pipeline):
        # Equivalent expressions should have high Jaccard
        out_a = pipeline.encode_math_only("x + 2")
        out_b = pipeline.encode_math_only("2 + x")
        mt_a  = set(t for t in out_a.tokens if not t.startswith("["))
        mt_b  = set(t for t in out_b.tokens if not t.startswith("["))
        jac   = _jaccard(mt_a, mt_b)
        assert jac > 0.5    # should be near 1.0 due to canonicalization

    def test_run_standard_small(self, comp):
        # Run just 3 expressions to keep test fast
        for expr in STANDARD_EXPRESSIONS[:3]:
            rec = comp._compare_one(expr, "standard")
            assert rec.mathtok.token_count > 0

    def test_deep_nesting_depth_increases(self, comp, pipeline):
        flat    = pipeline.encode_math_only("x + 1")
        nested  = pipeline.encode_math_only("sin(cos((x+1)^2))")
        flat_d  = max((m.depth for m in flat.metadata    if m.depth >= 0), default=0)
        nest_d  = max((m.depth for m in nested.metadata  if m.depth >= 0), default=0)
        assert nest_d > flat_d


# ── Utility helpers ───────────────────────────────────────────────────────

class TestHelpers:
    def test_jaccard_identical(self):
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_jaccard_disjoint(self):
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_jaccard_partial(self):
        j = _jaccard({"a", "b"}, {"b", "c"})
        assert abs(j - 1/3) < 1e-9

    def test_mean_empty(self):
        assert _mean([]) == 0.0

    def test_mean_values(self):
        assert abs(_mean([1.0, 2.0, 3.0]) - 2.0) < 1e-9

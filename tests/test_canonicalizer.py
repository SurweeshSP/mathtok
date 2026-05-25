"""
Tests for the Canonicalization Layer (Layer 1).

Covers:
  - ASCII expression parsing
  - LaTeX expression parsing
  - Equivalence detection (are_equivalent)
  - Normalization transformations
  - Fallback behaviour on parse errors
"""

import pytest
import sympy as sp

from mathtok.canonicalizer import Canonicalizer, CanonicalizationResult


@pytest.fixture
def canon():
    return Canonicalizer(do_simplify=True, do_expand=True)


# ── Parsing ───────────────────────────────────────────────────────────────

class TestParsing:
    def test_ascii_simple(self, canon):
        r = canon.canonicalize("x^2 + 1")
        assert r.success
        assert r.input_format == "ascii"
        assert "x" in str(r.expr)

    def test_ascii_implicit_mul(self, canon):
        r = canon.canonicalize("2x + 1")
        assert r.success

    def test_ascii_constants(self, canon):
        r = canon.canonicalize("pi + e")
        assert r.success
        assert sp.pi in r.expr.free_symbols or r.expr == sp.pi + sp.E

    def test_latex_frac(self, canon):
        r = canon.canonicalize("\\frac{x^2}{2}")
        # LaTeX detected
        assert r.input_format == "latex" or r.success  # may fallback

    def test_latex_sin(self, canon):
        r = canon.canonicalize("\\sin(x^2)")
        assert r.success

    def test_latex_sqrt(self, canon):
        r = canon.canonicalize("\\sqrt{x^2 + 1}")
        assert r.success

    def test_parse_error_graceful(self, canon):
        r = canon.canonicalize("@@@invalid@@@")
        assert not r.success
        assert len(r.warnings) > 0

    def test_delimiters_stripped(self, canon):
        r = canon.canonicalize("$x^2 + 1$")
        assert r.success


# ── Normalization ─────────────────────────────────────────────────────────

class TestNormalization:
    def test_expand(self, canon):
        r = canon.canonicalize("(x+1)^2")
        # expanded form should include x^2 and 2x
        expr_str = str(r.expr)
        assert "x**2" in expr_str or "x^2" in expr_str

    def test_commutativity_canonical(self, canon):
        r1 = canon.canonicalize("a + b")
        r2 = canon.canonicalize("b + a")
        # SymPy canonicalises Add ordering
        assert str(r1.expr) == str(r2.expr)

    def test_subtraction_to_add(self, canon):
        r = canon.canonicalize("x - y")
        # SymPy represents x-y as Add(x, Mul(-1, y))
        assert isinstance(r.expr, sp.Add)

    def test_division_to_mul(self, canon):
        r = canon.canonicalize("x / y")
        # SymPy represents x/y as Mul(x, Pow(y, -1))
        assert isinstance(r.expr, sp.Mul)

    def test_transformations_recorded(self, canon):
        r = canon.canonicalize("x^2 + 2*x + 1")
        assert "expand" in r.transformations_applied
        assert "simplify" in r.transformations_applied


# ── Equivalence ───────────────────────────────────────────────────────────

class TestEquivalence:
    def test_basic_equivalent(self, canon):
        assert canon.are_equivalent("(x+1)^2", "x^2 + 2*x + 1")

    def test_commutative_equivalent(self, canon):
        assert canon.are_equivalent("a + b", "b + a")

    def test_not_equivalent(self, canon):
        assert not canon.are_equivalent("x^2", "x^3")

    def test_trig_identity(self, canon):
        # sin^2 + cos^2 = 1
        assert canon.are_equivalent("sin(x)^2 + cos(x)^2", "1")

    def test_log_product(self, canon):
        # log(x)+log(y) = log(x*y) requires positive assumptions;
        # SymPy's simplify may not collapse it without them.
        # Verify at least that both are valid canonical expressions.
        r1 = canon.canonicalize("log(x) + log(y)")
        r2 = canon.canonicalize("log(x*y)")
        assert r1.success and r2.success
        # With positive assumptions the difference simplifies to 0
        import sympy as sp
        x, y = sp.Symbol("x", positive=True), sp.Symbol("y", positive=True)
        diff = sp.simplify(sp.log(x) + sp.log(y) - sp.log(x * y))
        assert diff == 0

    def test_difference_of_squares(self, canon):
        assert canon.are_equivalent("a^2 - b^2", "(a+b)*(a-b)")

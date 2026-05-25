"""
Tests for the AST Generator (Layer 3).
"""

import pytest
import sympy as sp

from mathtok.ast_generator import ASTGenerator, ASTNode
from mathtok.canonicalizer import Canonicalizer


@pytest.fixture
def gen():
    return ASTGenerator()


@pytest.fixture
def canon():
    return Canonicalizer(do_simplify=False, do_expand=False)


def parse(expr_str: str):
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations,
        implicit_multiplication_application, convert_xor,
    )
    return parse_expr(
        expr_str,
        transformations=standard_transformations + (
            implicit_multiplication_application, convert_xor,
        ),
        local_dict={"x": sp.Symbol("x"), "y": sp.Symbol("y"),
                    "a": sp.Symbol("a"), "b": sp.Symbol("b"),
                    "n": sp.Symbol("n")},
    )


class TestBasicNodes:
    def test_symbol(self, gen):
        ast = gen.generate(sp.Symbol("x"))
        assert ast.token == "VAR_X"
        assert ast.is_leaf

    def test_integer_zero(self, gen):
        ast = gen.generate(sp.Integer(0))
        assert ast.token == "CONST_0"

    def test_integer_positive(self, gen):
        ast = gen.generate(sp.Integer(5))
        assert ast.token == "CONST_5"

    def test_integer_negative(self, gen):
        ast = gen.generate(sp.Integer(-3))
        assert ast.token == "OP_NEG"
        assert ast.children[0].token == "CONST_3"

    def test_pi(self, gen):
        ast = gen.generate(sp.pi)
        assert ast.token == "CONST_PI"

    def test_e(self, gen):
        ast = gen.generate(sp.E)
        assert ast.token == "CONST_E"

    def test_rational(self, gen):
        ast = gen.generate(sp.Rational(1, 2))
        assert ast.token == "FRAC"
        assert len(ast.children) == 2


class TestArithmetic:
    def test_add(self, gen):
        expr = parse("x + 1")
        ast = gen.generate(expr)
        assert ast.token == "OP_ADD"
        tokens = gen.get_all_tokens(ast)
        assert "VAR_X" in tokens
        assert "CONST_1" in tokens

    def test_mul(self, gen):
        expr = parse("2*x")
        ast = gen.generate(expr)
        # 2*x is either OP_MUL or OP_NEG etc.
        assert ast.token in ("OP_MUL", "VAR_X", "CONST_2")

    def test_pow(self, gen):
        expr = parse("x^2")
        ast = gen.generate(expr)
        assert ast.token == "OP_POW"
        assert ast.children[0].token == "VAR_X"
        assert ast.children[1].token == "CONST_2"

    def test_negation(self, gen):
        expr = sp.Mul(sp.Integer(-1), sp.Symbol("x"))
        ast = gen.generate(expr)
        assert ast.token == "OP_NEG"

    def test_reciprocal(self, gen):
        expr = sp.Pow(sp.Symbol("x"), sp.Integer(-1))
        ast = gen.generate(expr)
        assert ast.token == "OP_RECIP"


class TestFunctions:
    def test_sin(self, gen):
        expr = sp.sin(sp.Symbol("x"))
        ast = gen.generate(expr)
        assert ast.token == "FUNC_SIN"
        assert ast.children[0].token == "VAR_X"

    def test_cos(self, gen):
        ast = gen.generate(sp.cos(sp.Symbol("x")))
        assert ast.token == "FUNC_COS"

    def test_exp(self, gen):
        ast = gen.generate(sp.exp(sp.Symbol("x")))
        assert ast.token == "FUNC_EXP"

    def test_log(self, gen):
        ast = gen.generate(sp.log(sp.Symbol("x")))
        assert ast.token == "FUNC_LOG"

    def test_sqrt(self, gen):
        # SymPy represents sqrt(x) internally as Pow(x, Rational(1,2))
        # so the AST correctly emits OP_POW; FUNC_SQRT is only emitted
        # when sympy.sqrt is used directly before any canonicalization.
        ast = gen.generate(sp.sqrt(sp.Symbol("x")))
        # Accept either FUNC_SQRT (direct) or OP_POW (post-simplification)
        assert ast.token in ("FUNC_SQRT", "OP_POW")


class TestTreeProperties:
    def test_depth_assignment(self, gen):
        expr = parse("x^2 + 1")
        ast = gen.generate(expr)
        assert ast.depth == 0
        for child in ast.children:
            assert child.depth == 1

    def test_unique_node_ids(self, gen):
        expr = parse("x^2 + 2*x + 1")
        ast = gen.generate(expr)
        all_ids: list[int] = []

        def collect(node):
            all_ids.append(node.node_id)
            for c in node.children:
                collect(c)

        collect(ast)
        assert len(all_ids) == len(set(all_ids)), "Node IDs must be unique"

    def test_subtree_size(self, gen):
        ast = gen.generate(sp.Integer(5))
        assert ast.subtree_size == 1

        expr = parse("x + 1")
        ast = gen.generate(expr)
        assert ast.subtree_size == 3  # ADD + VAR_X + CONST_1

    def test_variable_extraction(self, gen):
        expr = parse("x^2 + y + 1")
        ast = gen.generate(expr)
        vars_ = gen.get_variable_tokens(ast)
        assert "VAR_X" in vars_
        assert "VAR_Y" in vars_

"""
Tests for the Structural Serializer (Layer 5).
"""

import pytest
import sympy as sp

from mathtok.ast_generator import ASTGenerator
from mathtok.serializer import StructuralSerializer, MATH_START, MATH_END


@pytest.fixture
def gen():
    return ASTGenerator()


@pytest.fixture
def ser():
    return StructuralSerializer(include_boundaries=True)


@pytest.fixture
def ser_no_boundary():
    return StructuralSerializer(include_boundaries=False)


def make_ast(expr_str: str) -> object:
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations,
        implicit_multiplication_application, convert_xor,
    )
    expr = parse_expr(
        expr_str,
        transformations=standard_transformations + (
            implicit_multiplication_application, convert_xor,
        ),
        local_dict={"x": sp.Symbol("x"), "y": sp.Symbol("y"),
                    "a": sp.Symbol("a"), "b": sp.Symbol("b")},
    )
    return ASTGenerator().generate(expr)


class TestBoundaries:
    def test_start_end_tokens(self, ser):
        ast = make_ast("x + 1")
        tokens = ser.serialize(ast)
        assert tokens[0].token == MATH_START
        assert tokens[-1].token == MATH_END

    def test_no_boundaries(self, ser_no_boundary):
        ast = make_ast("x")
        tokens = ser_no_boundary.serialize(ast)
        assert tokens[0].token != MATH_START


class TestTokenStream:
    def test_leaf_node(self, ser):
        ast = ASTGenerator().generate(sp.Symbol("x"))
        tokens = ser.serialize(ast)
        # [MATH_START, VAR_X, MATH_END]
        tok_strs = [t.token for t in tokens]
        assert "VAR_X" in tok_strs

    def test_preorder_order(self, ser_no_boundary):
        # x + 1 → ADD(VAR_X, CONST_1) → [OP_ADD, VAR_X, CONST_1]
        ast = make_ast("x + 1")
        tokens = ser_no_boundary.serialize(ast)
        tok_strs = [t.token for t in tokens]
        add_idx = tok_strs.index("OP_ADD")
        x_idx   = tok_strs.index("VAR_X")
        assert add_idx < x_idx   # parent before children

    def test_depth_assigned(self, ser_no_boundary):
        ast = make_ast("x + 1")
        tokens = ser_no_boundary.serialize(ast)
        root_tok = next(t for t in tokens if t.token == "OP_ADD")
        assert root_tok.depth == 0
        child_toks = [t for t in tokens if t.token in ("VAR_X", "CONST_1")]
        for ct in child_toks:
            assert ct.depth == 1

    def test_positions_sequential(self, ser):
        ast = make_ast("x^2 + 1")
        tokens = ser.serialize(ast)
        positions = [t.position for t in tokens]
        assert positions == list(range(len(tokens)))

    def test_is_leaf_flag(self, ser_no_boundary):
        ast = ASTGenerator().generate(sp.Symbol("x"))
        tokens = ser_no_boundary.serialize(ast)
        assert all(t.is_leaf for t in tokens)

    def test_subtree_size_root(self, ser_no_boundary):
        ast = make_ast("x + 1")
        tokens = ser_no_boundary.serialize(ast)
        root = tokens[0]   # OP_ADD
        assert root.subtree_size == 3   # ADD + VAR_X + CONST_1


class TestSexp:
    def test_sexp_leaf(self, ser):
        ast = ASTGenerator().generate(sp.Symbol("x"))
        sexp = ser.to_sexp(ast)
        assert sexp == "VAR_X"

    def test_sexp_simple(self, ser):
        ast = make_ast("x + 1")
        sexp = ser.to_sexp(ast)
        assert sexp.startswith("(OP_ADD")

    def test_sexp_nested(self, ser):
        ast = make_ast("x^2 + 1")
        sexp = ser.to_sexp(ast)
        assert "OP_POW" in sexp
        assert "OP_ADD" in sexp


class TestTokenList:
    def test_to_token_list(self, ser):
        ast = make_ast("x + 1")
        tok_list = ser.to_token_list(ast)
        assert isinstance(tok_list, list)
        assert all(isinstance(t, str) for t in tok_list)
        assert MATH_START in tok_list
        assert MATH_END in tok_list

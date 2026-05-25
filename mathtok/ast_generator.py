"""
Layer 3: AST Generator

Converts a canonical SymPy expression into a typed ASTNode tree.
Each node carries:
  - token       : MathTok vocabulary string (e.g. "OP_ADD", "VAR_X")
  - sympy_expr  : the original SymPy subexpression
  - children    : ordered child ASTNodes
  - depth       : 0 = root
  - node_id     : unique integer assigned by DFS counter
  - parent_id   : -1 for root

The tree faithfully mirrors the SymPy internal representation while
mapping SymPy types onto the richer MathTok operator vocabulary.

Key design decisions
────────────────────
• Mul(-1, x) → OP_NEG(x)          (detect unary negation)
• Pow(x, -1) → OP_RECIP(x)        (detect reciprocal)
• Rational(p, q) → FRAC(p, q)     (explicit fraction node)
• Unknown functions → FUNC_<NAME>  (graceful fallback)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import sympy as sp
from sympy import (
    Add, Mul, Pow, Symbol, Integer, Rational, Float, Number,
    Abs, Derivative, Integral, Limit, Sum, Product,
    sin, cos, tan, asin, acos, atan, sinh, cosh, tanh,
    exp, log, sqrt, gamma, factorial, floor, ceiling, re, im,
    Eq, Ne, Lt, Gt, Le, Ge,
    S,
)

logger = logging.getLogger(__name__)


# ── ASTNode dataclass ──────────────────────────────────────────────────────

@dataclass
class ASTNode:
    """
    A node in the MathTok abstract syntax tree.

    Attributes
    ----------
    token : str
        MathTok vocabulary token, e.g. "OP_ADD", "VAR_X", "CONST_2".
    sympy_expr : Any
        Original SymPy (sub)expression for debugging / round-tripping.
    children : list[ASTNode]
        Ordered child nodes (left-to-right as in mathematical notation).
    depth : int
        Depth from the root (root = 0).
    node_id : int
        Unique integer ID assigned during tree construction.
    parent_id : int
        Parent node's ID; -1 for the root.
    """
    token: str
    sympy_expr: Any
    children: list[ASTNode] = field(default_factory=list)
    depth: int = 0
    node_id: int = -1
    parent_id: int = -1
    confidence: float = 1.0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def subtree_size(self) -> int:
        return 1 + sum(c.subtree_size for c in self.children)

    @property
    def height(self) -> int:
        if self.is_leaf:
            return 0
        return 1 + max(c.height for c in self.children)

    def __repr__(self) -> str:
        if self.children:
            return f"{self.token}({', '.join(repr(c) for c in self.children)})"
        return self.token

    def to_dict(self) -> dict:
        return {
            "token":      self.token,
            "node_id":    self.node_id,
            "parent_id":  self.parent_id,
            "depth":      self.depth,
            "is_leaf":    self.is_leaf,
            "subtree_size": self.subtree_size,
            "confidence": self.confidence,
            "children":   [c.to_dict() for c in self.children],
        }


# ── SymPy type → MathTok token mapping ────────────────────────────────────

_FUNC_MAP: dict[type, str] = {
    sin:        "FUNC_SIN",
    cos:        "FUNC_COS",
    tan:        "FUNC_TAN",
    asin:       "FUNC_ASIN",
    acos:       "FUNC_ACOS",
    atan:       "FUNC_ATAN",
    sinh:       "FUNC_SINH",
    cosh:       "FUNC_COSH",
    tanh:       "FUNC_TANH",
    exp:        "FUNC_EXP",
    log:        "FUNC_LOG",
    sqrt:       "FUNC_SQRT",
    Abs:        "OP_ABS",
    gamma:      "FUNC_GAMMA",
    factorial:  "FUNC_FACTORIAL",
    floor:      "FUNC_FLOOR",
    ceiling:    "FUNC_CEIL",
    re:         "FUNC_RE",
    im:         "FUNC_IM",
    Derivative: "OP_DERIV",
    Integral:   "OP_INT",
    Limit:      "OP_LIMIT",
    Sum:        "OP_SUM",
    Product:    "OP_PROD",
}

_REL_MAP: dict[type, str] = {
    Eq: "OP_EQ",
    Ne: "OP_NEQ",
    Lt: "OP_LT",
    Gt: "OP_GT",
    Le: "OP_LE",
    Ge: "OP_GE",
}

# Pre-defined variable tokens (name → token)
_VAR_MAP: dict[str, str] = {
    "x": "VAR_X",  "y": "VAR_Y",  "z": "VAR_Z",  "t": "VAR_T",
    "n": "VAR_N",  "k": "VAR_K",  "a": "VAR_A",  "b": "VAR_B",
    "c": "VAR_C",  "m": "VAR_M",  "i": "VAR_I",  "j": "VAR_J",
    "r": "VAR_R",  "s": "VAR_S",  "u": "VAR_U",  "v": "VAR_V",
    "w": "VAR_W",  "p": "VAR_P",  "q": "VAR_Q",  "l": "VAR_L",
    "f": "VAR_F",  "g": "VAR_G",  "h": "VAR_H",
    # Greek letters
    "theta":   "VAR_THETA",   "alpha":   "VAR_ALPHA",
    "beta":    "VAR_BETA",    "gamma":   "VAR_GAMMA_",
    "delta":   "VAR_DELTA",   "epsilon": "VAR_EPSILON",
    "zeta":    "VAR_ZETA",    "eta":     "VAR_ETA",
    "lambda":  "VAR_LAMBDA",  "mu":      "VAR_MU",
    "nu":      "VAR_NU",      "xi":      "VAR_XI",
    "rho":     "VAR_RHO",     "sigma":   "VAR_SIGMA",
    "tau":     "VAR_TAU",     "phi":     "VAR_PHI",
    "chi":     "VAR_CHI",     "psi":     "VAR_PSI",
    "omega":   "VAR_OMEGA",
}

# Small integer dedicated tokens (covers the vast majority of constants)
_INT_TOKENS: dict[int, str] = {i: f"CONST_{i}" for i in range(-10, 101)}


# ── ASTGenerator ──────────────────────────────────────────────────────────

class ASTGenerator:
    """
    Convert a canonical SymPy expression into a typed ASTNode tree.

    Usage
    -----
    >>> gen = ASTGenerator()
    >>> import sympy as sp
    >>> ast = gen.generate(sp.parse_expr("x**2 + 2*x + 1"))
    >>> print(ast)
    OP_ADD(OP_POW(VAR_X, CONST_2), OP_MUL(CONST_2, VAR_X), CONST_1)
    """

    def __init__(self, max_depth: int = 20) -> None:
        self.max_depth = max_depth
        self._counter: int = 0

    def generate(self, expr: sp.Expr) -> ASTNode:
        """
        Build the ASTNode tree for a SymPy expression.

        Parameters
        ----------
        expr : sp.Expr
            Canonical SymPy expression (output of Canonicalizer).

        Returns
        -------
        ASTNode
            Root of the typed AST.
        """
        self._counter = 0
        return self._visit(expr, depth=0, parent_id=-1)

    def get_all_tokens(self, root: ASTNode) -> list[str]:
        """Collect all tokens from a tree (preorder DFS)."""
        result: list[str] = []
        self._collect_tokens(root, result)
        return result

    def get_variable_tokens(self, root: ASTNode) -> set[str]:
        """Extract the set of variable tokens in the tree."""
        return {t for t in self.get_all_tokens(root) if t.startswith("VAR_")}

    def get_operator_tokens(self, root: ASTNode) -> set[str]:
        """Extract the set of operator/function tokens in the tree."""
        return {
            t for t in self.get_all_tokens(root)
            if t.startswith("OP_") or t.startswith("FUNC_") or t == "FRAC"
        }

    # ── Visitor dispatch ──────────────────────────────────────────────────

    def _visit(self, expr: sp.Expr, depth: int, parent_id: int) -> ASTNode:
        """Recursively build ASTNode for a SymPy expression."""
        nid = self._counter
        self._counter += 1

        if depth >= self.max_depth:
            return ASTNode("SUBTREE_TRUNCATED", expr, depth=depth, node_id=nid, parent_id=parent_id, confidence=0.0)

        # ── Special constants ─────────────────────────────────────────────
        if expr is sp.pi:
            return ASTNode("CONST_PI", expr, depth=depth, node_id=nid, parent_id=parent_id)
        if expr is sp.E:
            return ASTNode("CONST_E", expr, depth=depth, node_id=nid, parent_id=parent_id)
        if expr is sp.I:
            return ASTNode("CONST_I", expr, depth=depth, node_id=nid, parent_id=parent_id)
        if expr is sp.oo:
            return ASTNode("CONST_INF", expr, depth=depth, node_id=nid, parent_id=parent_id)
        if expr is sp.nan:
            return ASTNode("CONST_NAN", expr, depth=depth, node_id=nid, parent_id=parent_id)
        if expr == S.NegativeInfinity:
            return ASTNode("CONST_NEG_INF", expr, depth=depth, node_id=nid, parent_id=parent_id)

        # ── Integer ───────────────────────────────────────────────────────
        if isinstance(expr, Integer):
            val = int(expr)
            if val < 0:
                # Represent as OP_NEG(CONST_N)
                inner_token = _INT_TOKENS.get(-val, f"NUM_{-val}")
                inner = ASTNode(inner_token, -expr,
                                depth=depth + 1, node_id=self._counter, parent_id=nid)
                self._counter += 1
                return ASTNode("OP_NEG", expr, children=[inner],
                               depth=depth, node_id=nid, parent_id=parent_id)
            token = _INT_TOKENS.get(val, f"NUM_{val}")
            return ASTNode(token, expr, depth=depth, node_id=nid, parent_id=parent_id)

        # ── Rational (not integer) ────────────────────────────────────────
        if isinstance(expr, Rational):
            num_node = self._visit(Integer(expr.p), depth + 1, nid)
            den_node = self._visit(Integer(expr.q), depth + 1, nid)
            return ASTNode("FRAC", expr, children=[num_node, den_node],
                           depth=depth, node_id=nid, parent_id=parent_id)

        # ── Float ─────────────────────────────────────────────────────────
        if isinstance(expr, Float):
            safe = str(float(expr)).replace(".", "p").replace("-", "NEG")
            return ASTNode(f"FLOAT_{safe}", expr, depth=depth, node_id=nid, parent_id=parent_id)

        # ── Symbol ────────────────────────────────────────────────────────
        if isinstance(expr, Symbol):
            name = expr.name
            token = _VAR_MAP.get(name, f"VAR_{name.upper()}")
            return ASTNode(token, expr, depth=depth, node_id=nid, parent_id=parent_id)

        # ── Add ───────────────────────────────────────────────────────────
        if isinstance(expr, Add):
            children = [self._visit(a, depth + 1, nid) for a in expr.args]
            return ASTNode("OP_ADD", expr, children=children,
                           depth=depth, node_id=nid, parent_id=parent_id)

        # ── Mul ───────────────────────────────────────────────────────────
        if isinstance(expr, Mul):
            args = expr.args
            # Detect pure unary negation: Mul(-1, x)
            if len(args) == 2 and args[0] == Integer(-1):
                inner = self._visit(args[1], depth + 1, nid)
                return ASTNode("OP_NEG", expr, children=[inner],
                               depth=depth, node_id=nid, parent_id=parent_id)
            children = [self._visit(a, depth + 1, nid) for a in args]
            return ASTNode("OP_MUL", expr, children=children,
                           depth=depth, node_id=nid, parent_id=parent_id)

        # ── Pow ───────────────────────────────────────────────────────────
        if isinstance(expr, Pow):
            base_node = self._visit(expr.base, depth + 1, nid)
            # Detect reciprocal: x^{-1}
            if expr.exp == Integer(-1):
                return ASTNode("OP_RECIP", expr, children=[base_node],
                               depth=depth, node_id=nid, parent_id=parent_id)
            exp_node = self._visit(expr.exp, depth + 1, nid)
            return ASTNode("OP_POW", expr, children=[base_node, exp_node],
                           depth=depth, node_id=nid, parent_id=parent_id)

        # ── Known functions ───────────────────────────────────────────────
        expr_type = type(expr)
        if expr_type in _FUNC_MAP:
            token = _FUNC_MAP[expr_type]
            children = [self._visit(a, depth + 1, nid) for a in expr.args]
            return ASTNode(token, expr, children=children,
                           depth=depth, node_id=nid, parent_id=parent_id)

        # ── Relational ────────────────────────────────────────────────────
        if expr_type in _REL_MAP:
            token = _REL_MAP[expr_type]
            children = [self._visit(a, depth + 1, nid) for a in expr.args]
            return ASTNode(token, expr, children=children,
                           depth=depth, node_id=nid, parent_id=parent_id)

        # ── Generic fallback ──────────────────────────────────────────────
        cls_name = type(expr).__name__.upper()
        token = f"FUNC_{cls_name}"
        logger.debug("Unknown SymPy type %s → fallback token %s", type(expr).__name__, token)
        children = [self._visit(a, depth + 1, nid) for a in expr.args] if expr.args else []
        return ASTNode(token, expr, children=children,
                       depth=depth, node_id=nid, parent_id=parent_id, confidence=0.5)

    # ── Utilities ─────────────────────────────────────────────────────────

    def _collect_tokens(self, node: ASTNode, result: list[str]) -> None:
        result.append(node.token)
        for child in node.children:
            self._collect_tokens(child, result)

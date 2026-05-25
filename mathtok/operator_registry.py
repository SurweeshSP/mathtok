"""
Layer 4: Operator-Aware Semantic Registry

Every mathematical operator and function is assigned a rich metadata
record that captures its semantic role in mathematical computation.
This registry is the backbone of the structural token vocabulary.

Each OperatorMeta record encodes:
  - token        : unique string identifier in the MathTok vocabulary
  - sympy_type   : corresponding SymPy internal class name
  - arity        : number of operands (-1 = variadic)
  - precedence   : parsing binding strength (higher = tighter)
  - associativity: 'left' | 'right' | 'none'
  - semantic_role: high-level mathematical interpretation
  - latex_repr   : canonical LaTeX representation
  - ascii_repr   : ASCII fallback representation
  - category     : broad grouping for analysis
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional


# ── Data Model ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OperatorMeta:
    """Immutable semantic descriptor for a single MathTok operator token."""
    token: str
    sympy_type: str
    arity: int           # -1 = variadic
    precedence: int      # 0 = lowest binding
    associativity: str   # 'left' | 'right' | 'none'
    semantic_role: str
    latex_repr: str
    ascii_repr: str
    category: str        # 'arithmetic' | 'relational' | 'calculus' | 'function' | 'structural' | 'logic' | 'set' | 'geometry' | 'statistics'
    is_commutative: bool = False

    def to_dict(self) -> dict:
        return {
            "token":         self.token,
            "sympy_type":    self.sympy_type,
            "arity":         self.arity,
            "precedence":    self.precedence,
            "associativity": self.associativity,
            "semantic_role": self.semantic_role,
            "latex_repr":    self.latex_repr,
            "ascii_repr":    self.ascii_repr,
            "category":      self.category,
            "is_commutative": self.is_commutative,
        }


# ── Registry ──────────────────────────────────────────────────────────────

OPERATOR_REGISTRY: dict[str, OperatorMeta] = {

    # ── Arithmetic ──────────────────────────────────────────────────────
    "OP_ADD": OperatorMeta(
        token="OP_ADD", sympy_type="Add",
        arity=-1, precedence=1, associativity="left",
        semantic_role="aggregation",
        latex_repr="+", ascii_repr="+", category="arithmetic", is_commutative=True,
    ),
    "OP_MUL": OperatorMeta(
        token="OP_MUL", sympy_type="Mul",
        arity=-1, precedence=2, associativity="left",
        semantic_role="scaling",
        latex_repr="\\cdot", ascii_repr="*", category="arithmetic", is_commutative=True,
    ),
    "OP_POW": OperatorMeta(
        token="OP_POW", sympy_type="Pow",
        arity=2, precedence=4, associativity="right",
        semantic_role="recursive_growth",
        latex_repr="^", ascii_repr="**", category="arithmetic",
    ),
    "OP_NEG": OperatorMeta(
        token="OP_NEG", sympy_type="Mul",   # -x == Mul(-1, x) in SymPy
        arity=1, precedence=3, associativity="none",
        semantic_role="negation",
        latex_repr="-", ascii_repr="-", category="arithmetic",
    ),
    "OP_RECIP": OperatorMeta(
        token="OP_RECIP", sympy_type="Pow",  # x^{-1}
        arity=1, precedence=3, associativity="none",
        semantic_role="reciprocal",
        latex_repr="^{-1}", ascii_repr="**(-1)", category="arithmetic",
    ),
    "OP_ABS": OperatorMeta(
        token="OP_ABS", sympy_type="Abs",
        arity=1, precedence=5, associativity="none",
        semantic_role="magnitude",
        latex_repr="|\\cdot|", ascii_repr="abs", category="arithmetic",
    ),
    "FRAC": OperatorMeta(
        token="FRAC", sympy_type="Rational",
        arity=2, precedence=3, associativity="none",
        semantic_role="ratio",
        latex_repr="\\frac", ascii_repr="/", category="structural",
    ),

    # ── Relational ──────────────────────────────────────────────────────
    "OP_EQ": OperatorMeta(
        token="OP_EQ", sympy_type="Eq",
        arity=2, precedence=0, associativity="none",
        semantic_role="equality",
        latex_repr="=", ascii_repr="==", category="relational", is_commutative=True,
    ),
    "OP_NEQ": OperatorMeta(
        token="OP_NEQ", sympy_type="Ne",
        arity=2, precedence=0, associativity="none",
        semantic_role="inequality",
        latex_repr="\\neq", ascii_repr="!=", category="relational", is_commutative=True,
    ),
    "OP_LT": OperatorMeta(
        token="OP_LT", sympy_type="StrictLessThan",
        arity=2, precedence=0, associativity="none",
        semantic_role="strict_ordering",
        latex_repr="<", ascii_repr="<", category="relational",
    ),
    "OP_GT": OperatorMeta(
        token="OP_GT", sympy_type="StrictGreaterThan",
        arity=2, precedence=0, associativity="none",
        semantic_role="strict_ordering",
        latex_repr=">", ascii_repr=">", category="relational",
    ),
    "OP_LE": OperatorMeta(
        token="OP_LE", sympy_type="LessThan",
        arity=2, precedence=0, associativity="none",
        semantic_role="ordering",
        latex_repr="\\leq", ascii_repr="<=", category="relational",
    ),
    "OP_GE": OperatorMeta(
        token="OP_GE", sympy_type="GreaterThan",
        arity=2, precedence=0, associativity="none",
        semantic_role="ordering",
        latex_repr="\\geq", ascii_repr=">=", category="relational",
    ),

    # ── Calculus ────────────────────────────────────────────────────────
    "OP_DERIV": OperatorMeta(
        token="OP_DERIV", sympy_type="Derivative",
        arity=2, precedence=5, associativity="none",
        semantic_role="local_change",
        latex_repr="\\frac{d}{dx}", ascii_repr="diff", category="calculus",
    ),
    "OP_INT": OperatorMeta(
        token="OP_INT", sympy_type="Integral",
        arity=2, precedence=0, associativity="none",
        semantic_role="accumulation",
        latex_repr="\\int", ascii_repr="integrate", category="calculus",
    ),
    "OP_LIMIT": OperatorMeta(
        token="OP_LIMIT", sympy_type="Limit",
        arity=3, precedence=0, associativity="none",
        semantic_role="asymptotic_behavior",
        latex_repr="\\lim", ascii_repr="limit", category="calculus",
    ),
    "OP_SUM": OperatorMeta(
        token="OP_SUM", sympy_type="Sum",
        arity=2, precedence=0, associativity="none",
        semantic_role="discrete_accumulation",
        latex_repr="\\sum", ascii_repr="Sum", category="calculus",
    ),
    "OP_PROD": OperatorMeta(
        token="OP_PROD", sympy_type="Product",
        arity=2, precedence=0, associativity="none",
        semantic_role="discrete_scaling",
        latex_repr="\\prod", ascii_repr="Product", category="calculus",
    ),

    # ── Trigonometric Functions ─────────────────────────────────────────
    "FUNC_SIN": OperatorMeta(
        token="FUNC_SIN", sympy_type="sin",
        arity=1, precedence=5, associativity="none",
        semantic_role="periodic_oscillation",
        latex_repr="\\sin", ascii_repr="sin", category="function",
    ),
    "FUNC_COS": OperatorMeta(
        token="FUNC_COS", sympy_type="cos",
        arity=1, precedence=5, associativity="none",
        semantic_role="periodic_oscillation",
        latex_repr="\\cos", ascii_repr="cos", category="function",
    ),
    "FUNC_TAN": OperatorMeta(
        token="FUNC_TAN", sympy_type="tan",
        arity=1, precedence=5, associativity="none",
        semantic_role="periodic_ratio",
        latex_repr="\\tan", ascii_repr="tan", category="function",
    ),
    "FUNC_ASIN": OperatorMeta(
        token="FUNC_ASIN", sympy_type="asin",
        arity=1, precedence=5, associativity="none",
        semantic_role="inverse_periodic",
        latex_repr="\\arcsin", ascii_repr="asin", category="function",
    ),
    "FUNC_ACOS": OperatorMeta(
        token="FUNC_ACOS", sympy_type="acos",
        arity=1, precedence=5, associativity="none",
        semantic_role="inverse_periodic",
        latex_repr="\\arccos", ascii_repr="acos", category="function",
    ),
    "FUNC_ATAN": OperatorMeta(
        token="FUNC_ATAN", sympy_type="atan",
        arity=1, precedence=5, associativity="none",
        semantic_role="inverse_periodic",
        latex_repr="\\arctan", ascii_repr="atan", category="function",
    ),
    "FUNC_SINH": OperatorMeta(
        token="FUNC_SINH", sympy_type="sinh",
        arity=1, precedence=5, associativity="none",
        semantic_role="hyperbolic_oscillation",
        latex_repr="\\sinh", ascii_repr="sinh", category="function",
    ),
    "FUNC_COSH": OperatorMeta(
        token="FUNC_COSH", sympy_type="cosh",
        arity=1, precedence=5, associativity="none",
        semantic_role="hyperbolic_oscillation",
        latex_repr="\\cosh", ascii_repr="cosh", category="function",
    ),
    "FUNC_TANH": OperatorMeta(
        token="FUNC_TANH", sympy_type="tanh",
        arity=1, precedence=5, associativity="none",
        semantic_role="hyperbolic_ratio",
        latex_repr="\\tanh", ascii_repr="tanh", category="function",
    ),

    # ── Exponential / Logarithmic ────────────────────────────────────────
    "FUNC_EXP": OperatorMeta(
        token="FUNC_EXP", sympy_type="exp",
        arity=1, precedence=5, associativity="none",
        semantic_role="exponential_growth",
        latex_repr="e^", ascii_repr="exp", category="function",
    ),
    "FUNC_LOG": OperatorMeta(
        token="FUNC_LOG", sympy_type="log",
        arity=1, precedence=5, associativity="none",
        semantic_role="logarithmic_compression",
        latex_repr="\\ln", ascii_repr="log", category="function",
    ),
    "FUNC_LOG10": OperatorMeta(
        token="FUNC_LOG10", sympy_type="log",
        arity=1, precedence=5, associativity="none",
        semantic_role="logarithmic_compression",
        latex_repr="\\log_{10}", ascii_repr="log10", category="function",
    ),
    "FUNC_SQRT": OperatorMeta(
        token="FUNC_SQRT", sympy_type="sqrt",
        arity=1, precedence=5, associativity="none",
        semantic_role="root_extraction",
        latex_repr="\\sqrt", ascii_repr="sqrt", category="function",
    ),
    "FUNC_CBRT": OperatorMeta(
        token="FUNC_CBRT", sympy_type="cbrt",
        arity=1, precedence=5, associativity="none",
        semantic_role="root_extraction",
        latex_repr="\\sqrt[3]", ascii_repr="cbrt", category="function",
    ),

    # ── Special Functions ────────────────────────────────────────────────
    "FUNC_GAMMA": OperatorMeta(
        token="FUNC_GAMMA", sympy_type="gamma",
        arity=1, precedence=5, associativity="none",
        semantic_role="factorial_extension",
        latex_repr="\\Gamma", ascii_repr="gamma", category="function",
    ),
    "FUNC_FACTORIAL": OperatorMeta(
        token="FUNC_FACTORIAL", sympy_type="factorial",
        arity=1, precedence=6, associativity="none",
        semantic_role="combinatorial_growth",
        latex_repr="!", ascii_repr="factorial", category="function",
    ),
    "FUNC_FLOOR": OperatorMeta(
        token="FUNC_FLOOR", sympy_type="floor",
        arity=1, precedence=5, associativity="none",
        semantic_role="integer_rounding_down",
        latex_repr="\\lfloor\\rfloor", ascii_repr="floor", category="function",
    ),
    "FUNC_CEIL": OperatorMeta(
        token="FUNC_CEIL", sympy_type="ceiling",
        arity=1, precedence=5, associativity="none",
        semantic_role="integer_rounding_up",
        latex_repr="\\lceil\\rceil", ascii_repr="ceil", category="function",
    ),
    "FUNC_RE": OperatorMeta(
        token="FUNC_RE", sympy_type="re",
        arity=1, precedence=5, associativity="none",
        semantic_role="real_part",
        latex_repr="\\Re", ascii_repr="re", category="function",
    ),
    "FUNC_IM": OperatorMeta(
        token="FUNC_IM", sympy_type="im",
        arity=1, precedence=5, associativity="none",
        semantic_role="imaginary_part",
        latex_repr="\\Im", ascii_repr="im", category="function",
    ),

    # ── Logic ───────────────────────────────────────────────────────────
    "OP_AND": OperatorMeta(
        token="OP_AND", sympy_type="And",
        arity=-1, precedence=1, associativity="left",
        semantic_role="logical_conjunction",
        latex_repr="\\land", ascii_repr="and", category="logic", is_commutative=True,
    ),
    "OP_OR": OperatorMeta(
        token="OP_OR", sympy_type="Or",
        arity=-1, precedence=1, associativity="left",
        semantic_role="logical_disjunction",
        latex_repr="\\lor", ascii_repr="or", category="logic", is_commutative=True,
    ),
    "OP_NOT": OperatorMeta(
        token="OP_NOT", sympy_type="Not",
        arity=1, precedence=5, associativity="none",
        semantic_role="logical_negation",
        latex_repr="\\lnot", ascii_repr="not", category="logic",
    ),
    "OP_IMPLIES": OperatorMeta(
        token="OP_IMPLIES", sympy_type="Implies",
        arity=2, precedence=0, associativity="none",
        semantic_role="logical_implication",
        latex_repr="\\implies", ascii_repr="=>", category="logic",
    ),

    # ── Set Theory ──────────────────────────────────────────────────────
    "OP_UNION": OperatorMeta(
        token="OP_UNION", sympy_type="Union",
        arity=-1, precedence=2, associativity="left",
        semantic_role="set_union",
        latex_repr="\\cup", ascii_repr="U", category="set", is_commutative=True,
    ),
    "OP_INTERSECT": OperatorMeta(
        token="OP_INTERSECT", sympy_type="Intersection",
        arity=-1, precedence=2, associativity="left",
        semantic_role="set_intersection",
        latex_repr="\\cap", ascii_repr="intersect", category="set", is_commutative=True,
    ),
    "OP_IN": OperatorMeta(
        token="OP_IN", sympy_type="Contains",
        arity=2, precedence=0, associativity="none",
        semantic_role="set_membership",
        latex_repr="\\in", ascii_repr="in", category="set",
    ),
    "OP_SUBSET": OperatorMeta(
        token="OP_SUBSET", sympy_type="Subset",
        arity=2, precedence=0, associativity="none",
        semantic_role="subset",
        latex_repr="\\subset", ascii_repr="subset", category="set",
    ),

    # ── Geometry ────────────────────────────────────────────────────────
    "OP_ANGLE": OperatorMeta(
        token="OP_ANGLE", sympy_type="Angle",
        arity=1, precedence=5, associativity="none",
        semantic_role="geometric_angle",
        latex_repr="\\angle", ascii_repr="angle", category="geometry",
    ),
    "OP_PARALLEL": OperatorMeta(
        token="OP_PARALLEL", sympy_type="Parallel",
        arity=2, precedence=0, associativity="none",
        semantic_role="geometric_parallel",
        latex_repr="\\parallel", ascii_repr="||", category="geometry", is_commutative=True,
    ),
    "OP_PERP": OperatorMeta(
        token="OP_PERP", sympy_type="Perpendicular",
        arity=2, precedence=0, associativity="none",
        semantic_role="geometric_perpendicular",
        latex_repr="\\perp", ascii_repr="perp", category="geometry", is_commutative=True,
    ),

    # ── Statistics ──────────────────────────────────────────────────────
    "FUNC_MEAN": OperatorMeta(
        token="FUNC_MEAN", sympy_type="Mean",
        arity=-1, precedence=5, associativity="none",
        semantic_role="statistical_mean",
        latex_repr="\\mu", ascii_repr="mean", category="statistics",
    ),
    "FUNC_STDEV": OperatorMeta(
        token="FUNC_STDEV", sympy_type="StdDev",
        arity=-1, precedence=5, associativity="none",
        semantic_role="statistical_deviation",
        latex_repr="\\sigma", ascii_repr="stdev", category="statistics",
    ),
    "FUNC_VAR": OperatorMeta(
        token="FUNC_VAR", sympy_type="Variance",
        arity=-1, precedence=5, associativity="none",
        semantic_role="statistical_variance",
        latex_repr="\\sigma^2", ascii_repr="var", category="statistics",
    ),
}

INVERSE_PAIRS: dict[str, str] = {
    "FUNC_SIN": "FUNC_ASIN", "FUNC_ASIN": "FUNC_SIN",
    "FUNC_COS": "FUNC_ACOS", "FUNC_ACOS": "FUNC_COS",
    "FUNC_TAN": "FUNC_ATAN", "FUNC_ATAN": "FUNC_TAN",
    "FUNC_EXP": "FUNC_LOG", "FUNC_LOG": "FUNC_EXP",
    "OP_ADD": "OP_NEG", "OP_NEG": "OP_ADD",
}

# ── Derived Lookups ────────────────────────────────────────────────────────

# sympy class name → list of tokens (may be many-to-one, e.g. log)
SYMPY_TYPE_TO_TOKENS: dict[str, list[str]] = {}
for _tok, _meta in OPERATOR_REGISTRY.items():
    SYMPY_TYPE_TO_TOKENS.setdefault(_meta.sympy_type, []).append(_tok)

# Group tokens by category
OPERATOR_CATEGORIES: dict[str, list[str]] = {
    cat: [t for t, m in OPERATOR_REGISTRY.items() if m.category == cat]
    for cat in {"arithmetic", "relational", "calculus", "function", "structural", "logic", "set", "geometry", "statistics"}
}


# ── Public Helpers ─────────────────────────────────────────────────────────

def get_operator(token: str) -> Optional[OperatorMeta]:
    """Return OperatorMeta for a given token, or None."""
    return OPERATOR_REGISTRY.get(token)


def get_all_operator_tokens() -> List[str]:
    """Return all operator/function token strings."""
    return list(OPERATOR_REGISTRY.keys())


def get_by_category(category: str) -> List[str]:
    """Return all tokens in a given category."""
    return OPERATOR_CATEGORIES.get(category, [])

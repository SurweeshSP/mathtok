"""
Layer 1: Canonicalization Engine

Normalizes mathematically equivalent expressions so that structurally
similar inputs produce consistent token streams downstream.

Transformation pipeline
───────────────────────
  1. Format detection  — infer LaTeX vs ASCII from input heuristics
  2. Parse             — sympy.parsing.latex.parse_latex  OR
                         sympy.parsing.sympy_parser.parse_expr
  3. Expand            — distribute products/powers over sums
  4. Simplify          — apply algebraic identities (optional)
  5. Factor            — factorise if requested (off by default)
  6. Normalize sub/div — subtraction → Add(x, Mul(-1,y));
                         division   → Mul(x, Pow(y,-1))
                         (SymPy does this automatically internally)

Example
-------
  >>> c = Canonicalizer()
  >>> r = c.canonicalize("b + a")
  >>> print(r.canonical_str)   # "a + b"
  >>> c.are_equivalent("x^2 + 2*x + 1", "(x+1)^2")  # True
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional
import concurrent.futures

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

logger = logging.getLogger(__name__)

# Augmented ASCII transformation set
_ASCII_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# LaTeX detection markers — presence of any of these implies LaTeX input
_LATEX_MARKERS = (
    "\\frac", "\\sqrt", "\\int", "\\sum", "\\prod",
    "\\sin",  "\\cos",  "\\tan",  "\\log",  "\\ln",  "\\exp",
    "\\lim",  "\\cdot", "\\times", "\\infty",
    "\\alpha","\\beta", "\\gamma", "\\delta", "\\theta",
    "\\pi",   "\\sigma","\\mu",   "\\lambda","\\phi", "\\psi",
    "\\leq",  "\\geq",  "\\neq",  "\\in",   "\\subset",
    "{",                           # LaTeX grouping
)

# LaTeX math-mode delimiter pairs (outer, inner)
_LATEX_DELIMITERS = [
    ("$$", "$$"),
    ("$",  "$"),
    ("\\[", "\\]"),
    ("\\(", "\\)"),
]

# Local symbol dictionary for ASCII parser
_LOCAL_DICT: dict[str, object] = {
    "x": sp.Symbol("x"), "y": sp.Symbol("y"), "z": sp.Symbol("z"),
    "t": sp.Symbol("t"), "n": sp.Symbol("n"), "k": sp.Symbol("k"),
    "a": sp.Symbol("a"), "b": sp.Symbol("b"), "c": sp.Symbol("c"),
    "m": sp.Symbol("m"), "r": sp.Symbol("r"), "s": sp.Symbol("s"),
    "u": sp.Symbol("u"), "v": sp.Symbol("v"), "w": sp.Symbol("w"),
    "p": sp.Symbol("p"), "q": sp.Symbol("q"),
    "e":  sp.E,
    "pi": sp.pi,
    "i":  sp.I,
}


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class CanonicalizationResult:
    """Output of the canonicalization stage."""
    original: str
    expr: sp.Expr
    canonical_str: str
    input_format: str                          # 'latex' | 'ascii'
    transformations_applied: list[str] = field(default_factory=list)
    warnings: list[str]           = field(default_factory=list)
    success: bool = True

    def __repr__(self) -> str:
        return (
            f"CanonicalizationResult("
            f"fmt={self.input_format!r}, "
            f"canonical={self.canonical_str!r}, "
            f"ok={self.success})"
        )


# ── Main class ────────────────────────────────────────────────────────────

class Canonicalizer:
    """
    Canonicalize mathematical expressions (LaTeX or ASCII) via SymPy.

    Parameters
    ----------
    do_simplify : bool
        Apply sympy.simplify().  Recommended ON (may be slow for complex exprs).
    do_expand : bool
        Apply sympy.expand() before simplify.
    do_factor : bool
        Apply sympy.factor() as an alternative to expand+simplify.
    sort_operands : bool
        SymPy sorts Add/Mul operands canonically by default; flag kept for
        documentation clarity.
    """

    def __init__(
        self,
        do_simplify: bool = True,
        do_expand:   bool = True,
        do_factor:   bool = False,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.do_simplify = do_simplify
        self.do_expand   = do_expand
        self.do_factor   = do_factor
        self.timeout_seconds = timeout_seconds
        
        # Simple LRU cache setup
        self._cache: dict[str, CanonicalizationResult] = {}
        self._max_cache_size = 512

    # ── Public API ────────────────────────────────────────────────────────

    def canonicalize(self, expression: str) -> CanonicalizationResult:
        """
        Canonicalize a raw mathematical expression string with LRU caching.
        """
        expression = expression.strip()
        
        if expression in self._cache:
            return self._cache[expression]
            
        result = self._canonicalize_impl(expression)
        
        # Cache management
        if len(self._cache) >= self._max_cache_size:
            # Pop the oldest item (first inserted in Python 3.7+ dict)
            self._cache.pop(next(iter(self._cache)))
        self._cache[expression] = result
        
        return result

    def _canonicalize_impl(self, expression: str) -> CanonicalizationResult:
        """Internal canonicalize implementation without caching."""
        fmt, expr, warnings = self._parse(expression)
        applied: list[str] = [f"parse_{fmt}"]

        if expr is None:
            return CanonicalizationResult(
                original=expression,
                expr=sp.Symbol("PARSE_ERROR"),
                canonical_str="PARSE_ERROR",
                input_format=fmt,
                transformations_applied=applied,
                warnings=warnings,
                success=False,
            )

        # ── Normalization pipeline ────────────────────────────────────────
        if self.do_expand:
            expr, applied, warnings = _safe_apply(
                sp.expand, expr, "expand", applied, warnings, self.timeout_seconds
            )

        if self.do_simplify:
            expr, applied, warnings = _safe_apply(
                sp.simplify, expr, "simplify", applied, warnings, self.timeout_seconds
            )

        if self.do_factor:
            expr, applied, warnings = _safe_apply(
                sp.factor, expr, "factor", applied, warnings, self.timeout_seconds
            )

        # Subtraction/division normalization is automatic in SymPy's
        # internal representation (Add/Mul/Pow nodes).
        applied.append("normalize_sub_div")

        return CanonicalizationResult(
            original=expression,
            expr=expr,
            canonical_str=str(expr),
            input_format=fmt,
            transformations_applied=applied,
            warnings=warnings,
            success=True,
        )

    def are_equivalent(self, expr_a: str, expr_b: str) -> bool:
        """
        Return True iff two expressions are mathematically equivalent.

        Used for the Canonical Consistency Score (CCS) metric.
        """
        try:
            ra = self.canonicalize(expr_a)
            rb = self.canonicalize(expr_b)
            if not ra.success or not rb.success:
                return False
            diff = sp.simplify(ra.expr - rb.expr)
            return diff == 0
        except Exception as exc:
            logger.debug("are_equivalent failed: %s", exc)
            return False

    def batch_canonicalize(
        self, expressions: list[str]
    ) -> list[CanonicalizationResult]:
        """Canonicalize a list of expressions."""
        return [self.canonicalize(e) for e in expressions]

    # ── Parsing ───────────────────────────────────────────────────────────

    def _parse(
        self, expression: str
    ) -> tuple[str, Optional[sp.Expr], list[str]]:
        warnings: list[str] = []
        fmt = _detect_format(expression)
        cleaned = _strip_delimiters(expression)

        if fmt == "latex":
            expr = _parse_latex(cleaned, warnings)
            if expr is not None:
                return "latex", expr, warnings
            warnings.append("LaTeX parse failed — falling back to ASCII parser.")

        expr = _parse_ascii(cleaned, warnings)
        if expr is not None:
            return "ascii", expr, warnings

        return fmt, None, warnings


# ── Module-level helpers ───────────────────────────────────────────────────

def _detect_format(expression: str) -> str:
    """Heuristically decide if input is LaTeX or ASCII."""
    for marker in _LATEX_MARKERS:
        if marker in expression:
            return "latex"
    s = expression.strip()
    if s.startswith("$") or s.startswith("\\(") or s.startswith("\\["):
        return "latex"
    return "ascii"


def _strip_delimiters(expression: str) -> str:
    """Remove outer LaTeX math-mode delimiters."""
    s = expression.strip()
    for open_d, close_d in _LATEX_DELIMITERS:
        if s.startswith(open_d) and s.endswith(close_d) and len(s) > len(open_d) + len(close_d):
            return s[len(open_d):-len(close_d)].strip()
    return s


def _parse_latex(expression: str, warnings: list[str]) -> Optional[sp.Expr]:
    try:
        from sympy.parsing.latex import parse_latex  # antlr4 required
        return parse_latex(expression)
    except ImportError:
        warnings.append(
            "sympy.parsing.latex unavailable (install antlr4-python3-runtime==4.11.1)."
        )
        return None
    except Exception as exc:
        warnings.append(f"LaTeX parse error: {exc}")
        return None


def _parse_ascii(expression: str, warnings: list[str]) -> Optional[sp.Expr]:
    try:
        return parse_expr(
            expression,
            local_dict=_LOCAL_DICT,
            transformations=_ASCII_TRANSFORMS,
        )
    except Exception as exc:
        warnings.append(f"ASCII parse error: {exc}")
        return None


def _safe_apply(
    fn,
    expr: sp.Expr,
    name: str,
    applied: list[str],
    warnings: list[str],
    timeout_seconds: float = 5.0,
) -> tuple[sp.Expr, list[str], list[str]]:
    """Apply a SymPy transformation safely, catching all exceptions and timing out."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, expr)
        try:
            result = future.result(timeout=timeout_seconds)
            applied.append(name)
            return result, applied, warnings
        except concurrent.futures.TimeoutError:
            warnings.append(f"{name} timed out after {timeout_seconds}s")
            return expr, applied, warnings
        except Exception as exc:
            warnings.append(f"{name} failed: {exc}")
            return expr, applied, warnings

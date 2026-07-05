"""
MathTok Evaluation Metrics

Implements the five core metrics for evaluating structural tokenization
quality, as described in the MathTok paper:

  SCR  — Structural Compression Ratio
  CCS  — Canonical Consistency Score
  OPS  — Operator Preservation Score
  TS   — Token Stability
  TDF  — Tree Depth Fidelity

Each metric is self-contained and operates on TokenizedOutput objects
or lists of token strings, enabling easy integration into benchmark runs.

Baseline comparisons are supported for:
  - GPT-2 tokenizer (character-level BPE)
  - SentencePiece unigram
  - Character-level tokenization
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ── Metric result container ───────────────────────────────────────────────

@dataclass
class MetricResult:
    """Holds the value and supporting statistics for one metric."""
    name:        str
    value:       float
    description: str
    details:     dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"{self.name}: {self.value:.4f}  ({self.description})"


@dataclass
class EvaluationReport:
    """Full report across all five MathTok metrics."""
    scr:  MetricResult
    ccs:  MetricResult
    ops:  MetricResult
    ts:   MetricResult
    tdf:  MetricResult
    num_examples: int = 0

    def summary(self) -> str:
        lines = [
            f"{'='*60}",
            f"  MathTok Evaluation Report  (n={self.num_examples})",
            f"{'='*60}",
            f"  {self.scr}",
            f"  {self.ccs}",
            f"  {self.ops}",
            f"  {self.ts}",
            f"  {self.tdf}",
            f"{'='*60}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "num_examples": self.num_examples,
            "SCR": self.scr.value, "CCS": self.ccs.value,
            "OPS": self.ops.value, "TS":  self.ts.value,
            "TDF": self.tdf.value,
        }


# ── Metric 1: Structural Information (SI) & Canonical SCR ────────────────

def compute_structural_information(
    operator_nodes: int,
    tree_depth: int,
    parent_child_relations: int,
    function_scope: int,
    canonical_bonus: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
    epsilon: float = 1.0,
) -> float:
    """
    Computes SI(E) = alpha*O + beta*log(1+D) + gamma*R + delta*F + epsilon*C
    """
    return (
        alpha * operator_nodes
        + beta * math.log1p(tree_depth)
        + gamma * parent_child_relations
        + delta * function_scope
        + epsilon * canonical_bonus
    )


def structural_compression_ratio(
    si_values: list[float],
    token_counts: list[int],
) -> MetricResult:
    """
    SCR = SI(E) / |T(E)|

    Measures how efficiently the structural token stream represents the
    information content relative to the token count.
    Higher SCR = denser structural information per token.
    """
    assert len(si_values) == len(token_counts), "Length mismatch"
    ratios = []
    for si, tc in zip(si_values, token_counts):
        ratios.append(si / tc if tc > 0 else 0.0)

    mean_scr = sum(ratios) / len(ratios) if ratios else 0.0
    return MetricResult(
        name="SCR",
        value=mean_scr,
        description="Structural Compression Ratio (SI / tokens); higher = denser",
        details={
            "min": min(ratios) if ratios else 0,
            "max": max(ratios) if ratios else 0,
            "std": _std(ratios),
            "n":   len(ratios),
        },
    )


# ── Metric 2: Canonical Consistency Score (CCS) ──────────────────────────

def canonical_consistency_score(
    equivalent_pairs: list[tuple[str, str]],
    tokenize_fn: Callable[[str], list[str]],
) -> MetricResult:
    """
    CCS = Equivalent forms mapped identically / Total equivalent forms

    Measures the fraction of mathematically equivalent expressions that are
    tokenized into identically identical sets of tokens (excluding boundary tokens).
    CCS -> 1.0 means perfect consistency.

    Parameters
    ----------
    equivalent_pairs : list of (expr_A, expr_B) that are mathematically equal
    tokenize_fn      : function str → list[str] (the tokenizer under test)
    """
    identical_count = 0
    total = len(equivalent_pairs)
    
    for expr_a, expr_b in equivalent_pairs:
        try:
            toks_a = set(tokenize_fn(expr_a))
            toks_b = set(tokenize_fn(expr_b))
            # Remove boundary tokens
            toks_a = {t for t in toks_a if not t.startswith("[")}
            toks_b = {t for t in toks_b if not t.startswith("[")}
            
            if toks_a == toks_b:
                identical_count += 1
        except Exception as exc:
            logger.debug("CCS: failed on pair (%s, %s): %s", expr_a[:30], expr_b[:30], exc)

    ccs_value = identical_count / total if total > 0 else 0.0
    return MetricResult(
        name="CCS",
        value=ccs_value,
        description="Canonical Consistency Score — Fraction of equivalent forms mapped identically",
        details={"identical_count": identical_count, "total": total},
    )



# ── Metric 3: Operator Preservation Score (OPS) ──────────────────────────

def operator_preservation_score(
    expressions: list[str],
    tokenize_fn: Callable[[str], list[str]],
    expected_operators: Optional[list[set[str]]] = None,
) -> MetricResult:
    """
    OPS = fraction of expressions where all expected operator tokens appear.

    If expected_operators is not provided, we auto-detect expected operators
    from simple heuristics on the raw expression string.

    Parameters
    ----------
    expressions        : list of raw expression strings
    tokenize_fn        : str → list[str]
    expected_operators : optional list of sets of expected operator tokens
    """
    _OP_HEURISTICS: dict[str, str] = {
        "+": "OP_ADD",  "*": "OP_MUL",  "^": "OP_POW",  "**": "OP_POW",
        "/": "FRAC",    "sin": "FUNC_SIN", "cos": "FUNC_COS",
        "tan": "FUNC_TAN", "log": "FUNC_LOG", "exp": "FUNC_EXP",
        "sqrt": "FUNC_SQRT", "diff": "OP_DERIV", "integrate": "OP_INT",
        "lim": "OP_LIMIT", "sum": "OP_SUM", "factorial": "FUNC_FACTORIAL",
    }

    preserved = 0
    total     = 0

    for i, expr in enumerate(expressions):
        if expected_operators is not None:
            expected = expected_operators[i]
        else:
            # Heuristic: derive expected operators from raw expression
            expected = set()
            expr_lower = expr.lower()
            for key, op_tok in _OP_HEURISTICS.items():
                if key in expr_lower:
                    expected.add(op_tok)

        if not expected:
            continue   # skip if we can't determine expected operators

        try:
            tokens = set(tokenize_fn(expr))
        except Exception:
            tokens = set()

        if expected.issubset(tokens):
            preserved += 1
        total += 1

    ops_value = preserved / total if total > 0 else 1.0
    return MetricResult(
        name="OPS",
        value=ops_value,
        description="Operator Preservation Score — % of expressions with all expected ops (higher is better)",
        details={"preserved": preserved, "total": total},
    )


# ── Metric 4: Token Stability (TS) ───────────────────────────────────────

def token_stability(
    expression_groups: list[list[str]],
    tokenize_fn: Callable[[str], list[str]],
) -> MetricResult:
    """
    TS = 1 - mean( CoV(token_count) )  where CoV = std/mean.

    Measures how stable the token count is across syntactic rewritings
    of the same expression.  TS → 1.0 means perfectly stable.

    Parameters
    ----------
    expression_groups : list of groups; each group = rewritings of one expr
    tokenize_fn       : str → list[str]
    """
    covs = []
    for group in expression_groups:
        lengths = []
        for expr in group:
            try:
                lengths.append(len(tokenize_fn(expr)))
            except Exception:
                lengths.append(0)
        if len(lengths) < 2 or sum(lengths) == 0:
            continue
        mu  = sum(lengths) / len(lengths)
        std = _std(lengths)
        cov = std / mu if mu > 0 else 0.0
        covs.append(cov)

    mean_cov = sum(covs) / len(covs) if covs else 0.0
    ts_value = max(0.0, 1.0 - mean_cov)
    return MetricResult(
        name="TS",
        value=ts_value,
        description="Token Stability — 1 - CoV(token count across rewritings) (higher is better)",
        details={"mean_cov": mean_cov, "n_groups": len(covs)},
    )


# ── Metric 5: Tree Depth Fidelity (TDF) ──────────────────────────────────

def tree_depth_fidelity(
    expressions: list[str],
    tokenize_fn_with_meta: Callable,      # returns TokenizedOutput
    expected_depth_fn: Optional[Callable] = None,
) -> MetricResult:
    """
    TDF = 1 - mean( |actual_max_depth - expected_max_depth| / expected_max_depth )

    Measures how accurately the metadata captures the true tree depth.
    Relies on metadata.depth fields being correctly computed.

    Parameters
    ----------
    expressions           : list of expression strings
    tokenize_fn_with_meta : pipeline.encode() or equivalent
    expected_depth_fn     : optional callable(expr) → int for ground-truth depth
                            If None, uses sympy-computed depth as ground truth.
    """
    errors = []

    for expr in expressions:
        try:
            out = tokenize_fn_with_meta(expr)
            if not out.metadata:
                continue
            actual_depth = max((m.depth for m in out.metadata if m.depth >= 0), default=0)

            if expected_depth_fn is not None:
                expected_depth = expected_depth_fn(expr)
            else:
                # Use AST subtree height from first canon_result as ground truth
                if out.canon_results and out.canon_results[0].success:
                    import sympy as sp
                    expr_tree = out.canon_results[0].expr
                    expected_depth = _sympy_depth(expr_tree)
                else:
                    continue

            if expected_depth == 0:
                errors.append(0.0)
            else:
                rel_err = abs(actual_depth - expected_depth) / expected_depth
                errors.append(min(rel_err, 1.0))
        except Exception as exc:
            logger.debug("TDF: error on %s: %s", expr[:30], exc)
            errors.append(1.0)

    mean_err = sum(errors) / len(errors) if errors else 0.0
    tdf_value = max(0.0, 1.0 - mean_err)
    return MetricResult(
        name="TDF",
        value=tdf_value,
        description="Tree Depth Fidelity — accuracy of depth metadata vs ground truth (higher is better)",
        details={"mean_relative_error": mean_err, "n": len(errors)},
    )


# ── Baseline comparators ──────────────────────────────────────────────────

def tokenize_character_level(expr: str) -> list[str]:
    """Character-level tokenizer baseline."""
    return list(expr)


def make_gpt2_tokenizer():
    """Return a GPT-2 tokenizer as a baseline (requires transformers)."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        return lambda text: tok.tokenize(text)
    except Exception:
        logger.warning("GPT-2 tokenizer not available; using character baseline.")
        return tokenize_character_level


def make_sentencepiece_tokenizer(model_path: str):
    """Return a SentencePiece tokenizer baseline."""
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=model_path)
        return lambda text: sp.encode(text, out_type=str)
    except Exception:
        logger.warning("SentencePiece not available.")
        return tokenize_character_level


# ── Utility helpers ───────────────────────────────────────────────────────

def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu  = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _sympy_depth(expr) -> int:
    """Compute tree depth of a SymPy expression."""
    if not expr.args:
        return 0
    return 1 + max(_sympy_depth(a) for a in expr.args)

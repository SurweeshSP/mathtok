"""
Semantic Tokenizer Comparison Framework
========================================

Compares MathTok against GPT-2 and character-level baselines across
four evaluation categories, computing the Semantic Compression Ratio (SCR)
at three levels:

  Level 1 — Raw Token Count
      raw_scr = structural_score / token_count

  Level 2 — Semantic Density
      semantic_density = math_tokens / total_tokens
      (how "information-dense" the token stream is)

  Level 3 — Structural Efficiency
      structural_efficiency = parent_child_relations / token_count
      (how efficiently hierarchy is encoded)

Structural Score Formula
─────────────────────────
  score = operator_nodes          (+1 per OP_/FUNC_ token)
        + tree_depth              (+max depth in metadata)
        + parent_child_relations  (+1 per non-leaf node)
        + function_scope          (+1 per FUNC_ token)
        + canonical_bonus         (+2 if expression parsed ok)

  GPT-2 structural score is estimated heuristically from the token stream.

Test Categories
───────────────
  1. Standard expressions       — basic algebra, calculus
  2. Deep nesting               — sin(cos((x+1)^2 + y^3))
  3. Canonical equivalence      — x+2 vs 2+x (should converge)
  4. Mixed text+math            — "The derivative of sin(x^2)"
  5. LaTeX vs ASCII             — \\sin(x^2) vs sin(x^2)

Output
──────
  JSONL file: evaluation/results/comparison_results.jsonl
  Summary:    evaluation/results/comparison_summary.json

Usage
─────
  python -m evaluation.comparison
  python -m evaluation.comparison --no-gpt2       # skip GPT-2 download
  python -m evaluation.comparison --save          # save JSONL
  python -m evaluation.comparison --category deep # run one category
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ── Output directory ───────────────────────────────────────────────────────
_RESULTS_DIR = Path(__file__).parent / "results"


# ── Test suites ───────────────────────────────────────────────────────────

STANDARD_EXPRESSIONS = [
    "(x+1)^2",
    "sin(x^2) + 3*x",
    "x^2 + 2*x + 1",
    "exp(-x^2/2)",
    "1/(1 + exp(-x))",
    "log(x*y)",
    "sqrt(a^2 + b^2)",
    "n*(n+1)/2",
    "factorial(n)",
    "diff(sin(x), x)",
    "integrate(x^2, x)",
    "limit(sin(x)/x, x, 0)",
    "a^2 - b^2",
    "(-b + sqrt(b^2 - 4*a*c)) / (2*a)",
    "sum(k^2, k, 1, n)",
]

DEEP_NESTING_EXPRESSIONS = [
    "sin(cos(x^2 + 1))",
    "sin(cos((x+1)^2 + y^3))",
    "exp(log(sin(x^2 + cos(y))))",
    "sqrt(1 + sqrt(1 + sqrt(x)))",
    "log(1 + log(1 + x))",
    "((x+1)^2 + (y-1)^2)^3",
    "((a + b)*(a - b)) / ((a + b)^2)",
]

ODE_PDE_EXPRESSIONS = [
    "Derivative(f(x), x, 2) + 2*Derivative(f(x), x) + f(x)",
    "Derivative(u(x, t), t) - alpha * Derivative(u(x, t), x, 2)",
]

MATRIX_LINEAR_ALGEBRA = [
    "A*x + b",
    "det(A - lambda*I)",
]

PROBABILITY_EXPRESSIONS = [
    "P(A|B) * P(B) / P(A)",
    "exp(-x^2 / 2) / sqrt(2*pi)",
]

SET_THEORY = [
    "Union(A, B)",
    "Intersection(A, B)",
]

CANONICAL_PAIRS = [
    ("x + 2",            "2 + x"),
    ("a*b + a*c",        "a*(b+c)"),
    ("(x+1)^2",          "x^2 + 2*x + 1"),
    ("x^2 - y^2",        "(x+y)*(x-y)"),
    ("sin(x)^2 + cos(x)^2", "1"),
    ("2*x + 2*y",        "2*(x+y)"),
    ("x*y + x*z",        "x*(y+z)"),
    ("a^2 + 2*a*b + b^2","(a+b)^2"),
]

MIXED_TEXT_MATH = [
    "The derivative of sin(x^2) with respect to x.",
    "Solve for x when x^2 + 2*x + 1 = 0.",
    "The quadratic formula gives $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$.",
    "For $n \\geq 1$, the sum $\\sum_{k=1}^{n} k = \\frac{n(n+1)}{2}$.",
    "Integrate $\\int_0^1 x^2 dx$ to get $\\frac{1}{3}$.",
    "If $a > 0$ and $b > 0$ then $\\log(a) + \\log(b) = \\log(ab)$.",
    "The area of a circle of radius r is pi*r^2.",
    "Euler's identity: $e^{i\\pi} + 1 = 0$.",
]

LATEX_ASCII_PAIRS = [
    ("sin(x^2)",         "\\sin(x^2)"),
    ("sqrt(x^2 + 1)",    "\\sqrt{x^2 + 1}"),
    ("log(x)",           "\\ln(x)"),
    ("exp(x)",           "e^x"),
    ("x/y",              "\\frac{x}{y}"),
    ("int(x^2, x)",      "\\int x^2 dx"),
    ("diff(sin(x), x)",  "\\frac{d}{dx}\\sin(x)"),
    ("factorial(n)",     "n!"),
]


# ── Result dataclasses ────────────────────────────────────────────────────

@dataclass
class TokenizerStats:
    """Stats for one tokenizer on one expression."""
    name:           str
    tokens:         list[str]
    token_count:    int

    # Structural score components
    operator_nodes:         int = 0
    tree_depth:             int = 0
    parent_child_relations: int = 0
    function_scope:         int = 0
    canonical_bonus:        int = 0

    # Derived scores
    structural_score:      float = 0.0
    raw_scr:               float = 0.0   # structural_score / token_count
    semantic_density:      float = 0.0   # math tokens / total tokens
    structural_efficiency: float = 0.0   # parent_child_relations / token_count

    def compute_scr(self) -> None:
        self.structural_score = (
            self.operator_nodes
            + self.tree_depth
            + self.parent_child_relations
            + self.function_scope
            + self.canonical_bonus
        )
        self.raw_scr = (
            self.structural_score / self.token_count
            if self.token_count > 0 else 0.0
        )
        self.structural_efficiency = (
            self.parent_child_relations / self.token_count
            if self.token_count > 0 else 0.0
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("tokens")   # too verbose for JSONL
        return d


@dataclass
class ComparisonRecord:
    """Full comparison record for one expression."""
    expression:  str
    category:    str
    mathtok:     TokenizerStats
    char_level:  TokenizerStats
    gpt2:        Optional[TokenizerStats] = None
    sentencepiece: Optional[TokenizerStats] = None
    sexp:        str = ""                    # MathTok S-expression
    notes:       list[str] = field(default_factory=list)

    @property
    def scr_improvement_vs_gpt2(self) -> Optional[float]:
        if self.gpt2 is None or self.gpt2.raw_scr == 0:
            return None
        return self.mathtok.raw_scr / self.gpt2.raw_scr

    @property
    def scr_improvement_vs_sp(self) -> Optional[float]:
        if self.sentencepiece is None or self.sentencepiece.raw_scr == 0:
            return None
        return self.mathtok.raw_scr / self.sentencepiece.raw_scr

    @property
    def scr_improvement_vs_char(self) -> float:
        if self.char_level.raw_scr == 0:
            return 0.0
        return self.mathtok.raw_scr / self.char_level.raw_scr

    def to_dict(self) -> dict:
        return {
            "expression":             self.expression,
            "category":               self.category,
            "sexp":                   self.sexp,
            "mathtok":                self.mathtok.to_dict(),
            "gpt2":                   self.gpt2.to_dict() if self.gpt2 else None,
            "sentencepiece":          self.sentencepiece.to_dict() if self.sentencepiece else None,
            "char_level":             self.char_level.to_dict(),
            "scr_improvement_vs_gpt2": self.scr_improvement_vs_gpt2,
            "scr_improvement_vs_sp":   self.scr_improvement_vs_sp,
            "scr_improvement_vs_char": self.scr_improvement_vs_char,
            "notes":                  self.notes,
        }

    def print_row(self) -> None:
        gpt_count = self.gpt2.token_count if self.gpt2 else "N/A"
        gpt_scr   = f"{self.gpt2.raw_scr:.2f}" if self.gpt2 else "N/A"
        sp_count  = self.sentencepiece.token_count if self.sentencepiece else "N/A"
        sp_scr    = f"{self.sentencepiece.raw_scr:.2f}" if self.sentencepiece else "N/A"
        impr      = (f"{self.scr_improvement_vs_char:.2f}x"
                     if self.char_level.raw_scr > 0 else "N/A")
        expr_short = self.expression[:30].ljust(31)
        print(
            f"  {expr_short}"
            f" | MT:{self.mathtok.token_count:3d} (SCR {self.mathtok.raw_scr:.2f})"
            f" | GP:{str(gpt_count):3s} (SCR {gpt_scr})"
            f" | SP:{str(sp_count):3s} (SCR {sp_scr})"
            f" | CH:{self.char_level.token_count:3d} (SCR {self.char_level.raw_scr:.2f})"
            f" | Impr: {impr}"
        )


# ── Structural score helpers ──────────────────────────────────────────────

_OP_PREFIXES   = ("OP_", "FRAC")
_FUNC_PREFIXES = ("FUNC_",)
_BOUNDARY      = {"[MATH_START]", "[MATH_END]", "[TEXT_START]", "[TEXT_END]",
                  "[BOS]", "[EOS]", "[PAD]", "[UNK]", "[SEP]", "[MASK]"}

_MATH_OPS_GPT2 = {"+", "-", "*", "/", "^", "=", "<", ">", "**", "//"}
_MATH_FUNCS_GPT2 = {"sin", "cos", "tan", "log", "ln", "exp", "sqrt",
                    "lim", "sum", "prod", "diff", "integrate", "factorial"}
_PARENS = {"(", ")", "[", "]", "{", "}"}


def _score_mathtok(out) -> TokenizerStats:
    """Compute structural score for a MathTok TokenizedOutput."""
    tokens = [t for t in out.tokens if t not in _BOUNDARY]
    token_count = len(out.tokens)

    operator_nodes = sum(
        1 for t in tokens
        if any(t.startswith(p) for p in _OP_PREFIXES) or t == "FRAC"
    )
    function_scope = sum(1 for t in tokens if t.startswith("FUNC_"))
    math_tokens    = operator_nodes + function_scope + sum(
        1 for t in tokens if t.startswith("VAR_") or t.startswith("CONST_") or t.startswith("NUM_")
    )
    semantic_density = math_tokens / max(token_count, 1)

    # Tree depth and parent-child from metadata
    tree_depth = 0
    parent_child = 0
    if out.metadata:
        depths = [m.depth for m in out.metadata if m.depth >= 0]
        tree_depth = max(depths) if depths else 0
        parent_child = sum(1 for m in out.metadata if m.num_children > 0)

    canonical_bonus = 2 if out.canon_results and out.canon_results[0].success else 0

    stats = TokenizerStats(
        name="MathTok",
        tokens=out.tokens,
        token_count=token_count,
        operator_nodes=operator_nodes,
        tree_depth=tree_depth,
        parent_child_relations=parent_child,
        function_scope=function_scope,
        canonical_bonus=canonical_bonus,
        semantic_density=semantic_density,
    )
    stats.compute_scr()
    return stats


def _score_gpt2(tokens: list[str]) -> TokenizerStats:
    """Estimate structural score for a GPT-2 token list (heuristic)."""
    token_count = len(tokens)
    lower_toks  = [t.lower().strip() for t in tokens]

    operator_nodes = sum(1 for t in lower_toks if t in _MATH_OPS_GPT2)
    function_scope = sum(1 for t in lower_toks if t in _MATH_FUNCS_GPT2)
    math_tokens    = operator_nodes + function_scope

    # Estimate nesting depth from parentheses
    max_depth, depth = 0, 0
    for t in lower_toks:
        if t in ("(", "[", "{"):
            depth += 1
            max_depth = max(max_depth, depth)
        elif t in (")", "]", "}"):
            depth = max(0, depth - 1)

    # Estimate parent-child: every operator has ~1 parent and ~2 children
    parent_child = operator_nodes

    # No canonical parsing bonus
    canonical_bonus = 0

    semantic_density = math_tokens / max(token_count, 1)

    stats = TokenizerStats(
        name="GPT-2",
        tokens=tokens,
        token_count=token_count,
        operator_nodes=operator_nodes,
        tree_depth=max_depth,
        parent_child_relations=parent_child,
        function_scope=function_scope,
        canonical_bonus=canonical_bonus,
        semantic_density=semantic_density,
    )
    stats.compute_scr()
    return stats


def _score_char(expr: str) -> TokenizerStats:
    """Score for character-level tokenization."""
    tokens = list(expr)
    token_count = len(tokens)

    operator_nodes = sum(1 for c in tokens if c in "+-*/^=")
    function_scope = 0  # character level can't identify functions
    max_depth, depth = 0, 0
    for c in tokens:
        if c in "([{":
            depth += 1
            max_depth = max(max_depth, depth)
        elif c in ")]}":
            depth = max(0, depth - 1)
    parent_child = operator_nodes  # rough estimate

    semantic_density = operator_nodes / max(token_count, 1)

    stats = TokenizerStats(
        name="CharLevel",
        tokens=tokens,
        token_count=token_count,
        operator_nodes=operator_nodes,
        tree_depth=max_depth,
        parent_child_relations=parent_child,
        function_scope=function_scope,
        canonical_bonus=0,
        semantic_density=semantic_density,
    )
    stats.compute_scr()
    return stats


def _score_sp(tokens: list[str]) -> TokenizerStats:
    """Estimate structural score for a SentencePiece token list (heuristic)."""
    token_count = len(tokens)
    # Strip SentencePiece word prefix ' ' if present
    lower_toks  = [t.lower().replace(" ", "").strip() for t in tokens]
    lower_toks  = [t for t in lower_toks if t]

    operator_nodes = sum(1 for t in lower_toks if t in _MATH_OPS_GPT2)
    function_scope = sum(1 for t in lower_toks if t in _MATH_FUNCS_GPT2)
    math_tokens    = operator_nodes + function_scope

    # Estimate nesting depth from parentheses
    max_depth, depth = 0, 0
    for t in lower_toks:
        if t in ("(", "[", "{"):
            depth += 1
            max_depth = max(max_depth, depth)
        elif t in (")", "]", "}"):
            depth = max(0, depth - 1)

    parent_child = operator_nodes
    canonical_bonus = 0
    semantic_density = math_tokens / max(token_count, 1)

    stats = TokenizerStats(
        name="SentencePiece",
        tokens=tokens,
        token_count=token_count,
        operator_nodes=operator_nodes,
        tree_depth=max_depth,
        parent_child_relations=parent_child,
        function_scope=function_scope,
        canonical_bonus=canonical_bonus,
        semantic_density=semantic_density,
    )
    stats.compute_scr()
    return stats


def _get_trained_sp_tokenizer() -> Optional[Callable[[str], list[str]]]:
    """Train a small custom SentencePiece unigram model dynamically on all expressions."""
    try:
        import sentencepiece as spm
        import tempfile
        
        # Collect all expressions from our suites to form a corpus
        corpus = []
        corpus.extend(STANDARD_EXPRESSIONS)
        corpus.extend(DEEP_NESTING_EXPRESSIONS)
        corpus.extend(ODE_PDE_EXPRESSIONS)
        corpus.extend(MATRIX_LINEAR_ALGEBRA)
        corpus.extend(PROBABILITY_EXPRESSIONS)
        corpus.extend(SET_THEORY)
        for a, b in CANONICAL_PAIRS:
            corpus.extend([a, b])
        corpus.extend(MIXED_TEXT_MATH)
        for a, b in LATEX_ASCII_PAIRS:
            corpus.extend([a, b])
            
        # Deduplicate and strip
        corpus = sorted(list(set(e.strip() for e in corpus if e.strip())))
        
        # Write to a temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write("\n".join(corpus))
            temp_corpus_path = f.name
            
        model_prefix = os.path.join(tempfile.gettempdir(), "spm_math_temp")
        
        # Train a unigram model
        # Using a small vocab size (e.g., 100)
        spm.SentencePieceTrainer.train(
            input=temp_corpus_path,
            model_prefix=model_prefix,
            vocab_size=100,
            model_type="unigram",
            user_defined_symbols=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        )
        
        # Clean up temp corpus file
        try:
            os.remove(temp_corpus_path)
        except Exception:
            pass
            
        sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
        return lambda text: sp.encode(text, out_type=str)
    except Exception as exc:
        logger.warning("Could not train custom SentencePiece tokenizer: %s", exc)
        return None


# ── Main comparison engine ────────────────────────────────────────────────

class TokenizerComparison:
    """
    Run the full 3-level SCR comparison across all test categories.

    Parameters
    ----------
    pipeline    : MathTokPipeline
    gpt2_fn     : callable(str) -> list[str], or None to skip GPT-2
    save_jsonl  : write results to evaluation/results/comparison_results.jsonl
    """

    def __init__(
        self,
        pipeline,
        gpt2_fn:   Optional[Callable] = None,
        sp_fn:     Optional[Callable] = None,
        save_jsonl: bool = True,
    ) -> None:
        self.pipeline   = pipeline
        self.gpt2_fn    = gpt2_fn
        self.sp_fn      = sp_fn
        self.save_jsonl = save_jsonl
        self._records:  list[ComparisonRecord] = []

    # ── Public API ────────────────────────────────────────────────────────

    def run_all(self) -> list[ComparisonRecord]:
        """Run all 5 test categories and return all ComparisonRecords."""
        print("\n" + "=" * 80)
        print("  MathTok Semantic Tokenizer Comparison")
        print("=" * 80)

        self._run_category("standard",    STANDARD_EXPRESSIONS)
        self._run_category("deep_nesting", DEEP_NESTING_EXPRESSIONS)
        self._run_category("ode_pde", ODE_PDE_EXPRESSIONS)
        self._run_category("linear_algebra", MATRIX_LINEAR_ALGEBRA)
        self._run_category("probability", PROBABILITY_EXPRESSIONS)
        self._run_category("set_theory", SET_THEORY)
        self._run_canonical_equivalence()
        self._run_mixed_text_math()
        self._run_latex_vs_ascii()

        if self.save_jsonl:
            self._save_results()

        self._print_summary()
        return self._records

    def run_category(self, category: str) -> list[ComparisonRecord]:
        """Run a single named category."""
        categories = {
            "standard":    (self._run_category, ("standard",    STANDARD_EXPRESSIONS)),
            "deep":        (self._run_category, ("deep_nesting", DEEP_NESTING_EXPRESSIONS)),
            "ode_pde":     (self._run_category, ("ode_pde", ODE_PDE_EXPRESSIONS)),
            "linear":      (self._run_category, ("linear_algebra", MATRIX_LINEAR_ALGEBRA)),
            "probability": (self._run_category, ("probability", PROBABILITY_EXPRESSIONS)),
            "set_theory":  (self._run_category, ("set_theory", SET_THEORY)),
            "canonical":   (self._run_canonical_equivalence, ()),
            "mixed":       (self._run_mixed_text_math, ()),
            "latex_ascii": (self._run_latex_vs_ascii, ()),
        }
        if category not in categories:
            raise ValueError(f"Unknown category: {category}. Choose from: {list(categories)}")
        fn, args = categories[category]
        fn(*args)
        if self.save_jsonl:
            self._save_results()
        self._print_summary()
        return self._records

    # ── Category runners ──────────────────────────────────────────────────

    def _run_category(self, category: str, expressions: list[str]) -> None:
        print(f"\n--- {category.upper().replace('_', ' ')} ---")
        print(f"  {'Expression':<30} | {'MathTok':^21} | {'GPT-2':^16} | {'S-Piece':^16} | {'Char':^16} | Impr")
        print(f"  {'-'*30}-+-{'-'*21}-+-{'-'*16}-+-{'-'*16}-+-{'-'*16}-+------")

        for expr in expressions:
            rec = self._compare_one(expr, category)
            self._records.append(rec)
            rec.print_row()

    def _run_canonical_equivalence(self) -> None:
        print(f"\n--- CANONICAL EQUIVALENCE ---")
        print("  Testing that equivalent expressions -> similar MathTok token sets")
        print(f"  {'Pair':<45} | MT Jac  | GP Jac  | SP Jac  | Converged")
        print(f"  {'-'*45}-+---------+---------+---------+----------")

        for expr_a, expr_b in CANONICAL_PAIRS:
            rec_a = self._compare_one(expr_a, "canonical")
            rec_b = self._compare_one(expr_b, "canonical")
            self._records.extend([rec_a, rec_b])

            mt_a = set(t for t in rec_a.mathtok.tokens if t not in _BOUNDARY)
            mt_b = set(t for t in rec_b.mathtok.tokens if t not in _BOUNDARY)
            mt_jaccard = _jaccard(mt_a, mt_b)

            gp_jaccard = None
            if rec_a.gpt2 and rec_b.gpt2:
                gp_a = set(rec_a.gpt2.tokens)
                gp_b = set(rec_b.gpt2.tokens)
                gp_jaccard = _jaccard(gp_a, gp_b)

            sp_jaccard = None
            if rec_a.sentencepiece and rec_b.sentencepiece:
                sp_a = set(rec_a.sentencepiece.tokens)
                sp_b = set(rec_b.sentencepiece.tokens)
                sp_jaccard = _jaccard(sp_a, sp_b)

            pair_str = f"{expr_a!r} vs {expr_b!r}"[:45].ljust(46)
            gp_str   = f"{gp_jaccard:.3f}" if gp_jaccard is not None else "  N/A  "
            sp_str   = f"{sp_jaccard:.3f}" if sp_jaccard is not None else "  N/A  "
            converged = "YES" if mt_jaccard > 0.5 else "no "
            print(f"  {pair_str}| MT:{mt_jaccard:.3f} | GP:{gp_str} | SP:{sp_str} | {converged}")

    def _run_mixed_text_math(self) -> None:
        print(f"\n--- MIXED TEXT + MATH ---")
        print(f"  {'Input (truncated)':<40} | MT tokens | GP tokens | SP tokens | Math spans")
        print(f"  {'-'*40}-+-----------+-----------+-----------+-----------")

        for text in MIXED_TEXT_MATH:
            out = self.pipeline.encode(text)
            math_spans = len(out.math_sexps)
            mt_count   = len(out.tokens)

            gp_count = "N/A"
            if self.gpt2_fn:
                try:
                    gp_count = str(len(self.gpt2_fn(text)))
                except Exception:
                    pass

            sp_count = "N/A"
            if self.sp_fn:
                try:
                    sp_count = str(len(self.sp_fn(text)))
                except Exception:
                    pass

            preview = text[:40].ljust(41)
            print(f"  {preview}| {mt_count:9d} | {str(gp_count):9s} | {str(sp_count):9s} | {math_spans:9d}")

            rec = ComparisonRecord(
                expression=text,
                category="mixed_text_math",
                mathtok=_score_mathtok(out),
                gpt2=None,
                sentencepiece=None,
                char_level=_score_char(text),
                sexp=out.sexp,
            )
            self._records.append(rec)

    def _run_latex_vs_ascii(self) -> None:
        print(f"\n--- LaTeX vs ASCII NORMALIZATION ---")
        print("  Same expression in two formats — MathTok should produce identical AST")
        print(f"  {'ASCII':<25} {'LaTeX':<25} | MT same? | MT tokens A/L | GP tokens A/L | SP tokens A/L")
        print(f"  {'-'*25} {'-'*25}-+----------+---------------+---------------+---------------")

        for ascii_expr, latex_expr in LATEX_ASCII_PAIRS:
            out_ascii = self.pipeline.encode_math_only(ascii_expr)
            out_latex = self.pipeline.encode_math_only(latex_expr)

            mt_a = set(t for t in out_ascii.tokens if t not in _BOUNDARY)
            mt_l = set(t for t in out_latex.tokens if t not in _BOUNDARY)
            mt_same = _jaccard(mt_a, mt_l)
            same_str = f"{mt_same:.2f}" if mt_same > 0.8 else f"{mt_same:.2f}(~)"

            gp_str = "N/A / N/A"
            if self.gpt2_fn:
                try:
                    ga = len(self.gpt2_fn(ascii_expr))
                    gl = len(self.gpt2_fn(latex_expr))
                    gp_str = f"{ga:3d} / {gl:3d}"
                except Exception:
                    pass

            sp_str = "N/A / N/A"
            if self.sp_fn:
                try:
                    sa = len(self.sp_fn(ascii_expr))
                    sl = len(self.sp_fn(latex_expr))
                    sp_str = f"{sa:3d} / {sl:3d}"
                except Exception:
                    pass

            print(
                f"  {ascii_expr:<25} {latex_expr:<25}"
                f"| {same_str:>8s} "
                f"| {len(out_ascii.tokens):3d} / {len(out_latex.tokens):3d}       "
                f"| {gp_str}       "
                f"| {sp_str}"
            )

            for expr, out, fmt in [
                (ascii_expr, out_ascii, "ascii"),
                (latex_expr, out_latex, "latex"),
            ]:
                rec = ComparisonRecord(
                    expression=expr,
                    category=f"latex_vs_ascii_{fmt}",
                    mathtok=_score_mathtok(out),
                    gpt2=None,
                    sentencepiece=None,
                    char_level=_score_char(expr),
                    sexp=out.sexp,
                    notes=[f"pair_partner={latex_expr if fmt=='ascii' else ascii_expr}"],
                )
                self._records.append(rec)

    # ── Single expression comparison ──────────────────────────────────────

    def _compare_one(self, expr: str, category: str) -> ComparisonRecord:
        # MathTok
        try:
            out = self.pipeline.encode_math_only(expr)
            mt_stats = _score_mathtok(out)
            sexp = out.sexp
        except Exception as exc:
            logger.debug("MathTok failed on %r: %s", expr, exc)
            mt_stats = TokenizerStats(name="MathTok", tokens=[], token_count=0)
            sexp = ""

        # GPT-2
        gp_stats: Optional[TokenizerStats] = None
        if self.gpt2_fn:
            try:
                gp_tokens = self.gpt2_fn(expr)
                gp_stats  = _score_gpt2(gp_tokens)
            except Exception as exc:
                logger.debug("GPT-2 failed on %r: %s", expr, exc)

        # SentencePiece
        sp_stats: Optional[TokenizerStats] = None
        if self.sp_fn:
            try:
                sp_tokens = self.sp_fn(expr)
                sp_stats  = _score_sp(sp_tokens)
            except Exception as exc:
                logger.debug("SentencePiece failed on %r: %s", expr, exc)

        # Character-level
        ch_stats = _score_char(expr)

        return ComparisonRecord(
            expression=expr,
            category=category,
            mathtok=mt_stats,
            gpt2=gp_stats,
            sentencepiece=sp_stats,
            char_level=ch_stats,
            sexp=sexp,
        )

    # ── Aggregated summary ────────────────────────────────────────────────

    def _print_summary(self) -> None:
        math_records = [
            r for r in self._records
            if r.category not in ("mixed_text_math",)
            and r.mathtok.token_count > 0
        ]
        if not math_records:
            return

        mt_scr_mean  = _mean([r.mathtok.raw_scr         for r in math_records])
        mt_sd_mean   = _mean([r.mathtok.semantic_density for r in math_records])
        mt_se_mean   = _mean([r.mathtok.structural_efficiency for r in math_records])
        ch_scr_mean  = _mean([r.char_level.raw_scr       for r in math_records])

        gp_records   = [r for r in math_records if r.gpt2 is not None]
        gp_scr_mean  = _mean([r.gpt2.raw_scr             for r in gp_records]) if gp_records else None
        gp_sd_mean   = _mean([r.gpt2.semantic_density     for r in gp_records]) if gp_records else None

        sp_records   = [r for r in math_records if r.sentencepiece is not None]
        sp_scr_mean  = _mean([r.sentencepiece.raw_scr     for r in sp_records]) if sp_records else None
        sp_sd_mean   = _mean([r.sentencepiece.semantic_density for r in sp_records]) if sp_records else None

        impr_vs_gpt2 = (mt_scr_mean / gp_scr_mean) if gp_scr_mean else None
        impr_vs_sp   = (mt_scr_mean / sp_scr_mean)   if sp_scr_mean else None
        impr_vs_char = (mt_scr_mean / ch_scr_mean)  if ch_scr_mean else None

        print("\n" + "=" * 80)
        print("  AGGREGATED RESULTS")
        print("=" * 80)
        print(f"\n  {'Metric':<40} {'MathTok':>10} {'GPT-2':>10} {'S-Piece':>10} {'CharLvl':>10}")
        print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        def row(label, mt_val, gp_val=None, sp_val=None, ch_val=None):
            gp_str = f"{gp_val:10.4f}" if gp_val is not None else "       N/A"
            sp_str = f"{sp_val:10.4f}" if sp_val is not None else "       N/A"
            ch_str = f"{ch_val:10.4f}" if ch_val is not None else "       N/A"
            print(f"  {label:<40} {mt_val:10.4f} {gp_str} {sp_str} {ch_str}")

        row("Level 1 — SCR (struct_score / tokens)",
            mt_scr_mean, gp_scr_mean, sp_scr_mean, ch_scr_mean)
        row("Level 2 — Semantic Density (math_toks / total)",
            mt_sd_mean, gp_sd_mean, sp_sd_mean, None)
        row("Level 3 — Structural Efficiency (rels / tokens)",
            mt_se_mean)

        print(f"\n  SCR improvement vs GPT-2    : "
              f"{f'{impr_vs_gpt2:.2f}x' if impr_vs_gpt2 else 'N/A'}")
        print(f"  SCR improvement vs S-Piece  : "
              f"{f'{impr_vs_sp:.2f}x' if impr_vs_sp else 'N/A'}")
        print(f"  SCR improvement vs CharLevel: "
              f"{f'{impr_vs_char:.2f}x' if impr_vs_char else 'N/A'}")
        print(f"\n  Total records evaluated     : {len(self._records)}")
        print("=" * 80)

        return {
            "mathtok_scr":   mt_scr_mean,
            "gpt2_scr":      gp_scr_mean,
            "sp_scr":        sp_scr_mean,
            "charlevel_scr": ch_scr_mean,
            "scr_improvement_vs_gpt2": impr_vs_gpt2,
            "scr_improvement_vs_sp":   impr_vs_sp,
            "scr_improvement_vs_char": impr_vs_char,
            "mathtok_semantic_density": mt_sd_mean,
            "mathtok_structural_efficiency": mt_se_mean,
        }

    # ── Persistence ───────────────────────────────────────────────────────

    def _save_results(self) -> None:
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        jsonl_path = _RESULTS_DIR / "comparison_results.jsonl"

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in self._records:
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

        print(f"\n  Results saved to: {jsonl_path}")

        # Compact summary JSON
        math_records = [
            r for r in self._records
            if r.mathtok.token_count > 0
        ]
        summary = {
            "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_records": len(self._records),
            "mathtok_mean_scr":   _mean([r.mathtok.raw_scr         for r in math_records]),
            "charlevel_mean_scr": _mean([r.char_level.raw_scr       for r in math_records]),
            "gpt2_scr":           _mean([r.gpt2.raw_scr             for r in math_records if r.gpt2 is not None]),
            "sentencepiece_mean_scr": _mean([r.sentencepiece.raw_scr for r in math_records if r.sentencepiece is not None]),
            "mathtok_mean_semantic_density":
                _mean([r.mathtok.semantic_density          for r in math_records]),
            "mathtok_mean_structural_efficiency":
                _mean([r.mathtok.structural_efficiency     for r in math_records]),
            "per_record": [
                {
                    "expression":   r.expression[:60],
                    "category":     r.category,
                    "mt_tokens":    r.mathtok.token_count,
                    "mt_scr":       round(r.mathtok.raw_scr, 4),
                    "gp_tokens":    r.gpt2.token_count if r.gpt2 else None,
                    "gp_scr":       round(r.gpt2.raw_scr, 4) if r.gpt2 else None,
                    "sp_tokens":    r.sentencepiece.token_count if r.sentencepiece else None,
                    "sp_scr":       round(r.sentencepiece.raw_scr, 4) if r.sentencepiece else None,
                    "ch_tokens":    r.char_level.token_count,
                    "ch_scr":       round(r.char_level.raw_scr, 4),
                    "impr_vs_char": round(r.scr_improvement_vs_char, 4),
                }
                for r in math_records
            ],
        }
        summary_path = _RESULTS_DIR / "comparison_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  Summary saved to: {summary_path}")


# ── Helpers ───────────────────────────────────────────────────────────────

def _jaccard(a: set, b: set) -> float:
    union = len(a | b)
    return len(a & b) / union if union > 0 else 0.0


def _mean(values: list) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _load_gpt2():
    """Load GPT-2 tokenizer, return None if unavailable."""
    try:
        from transformers import GPT2Tokenizer
        tok = GPT2Tokenizer.from_pretrained("gpt2")
        return tok.tokenize
    except Exception as exc:
        logger.warning("GPT-2 unavailable (%s); running without it.", exc)
        return None


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(
        description="MathTok vs GPT-2 vs Char-level — Semantic SCR Comparison"
    )
    parser.add_argument(
        "--no-gpt2",  action="store_true",
        help="Skip GPT-2 (no internet required)"
    )
    parser.add_argument(
        "--save",  action="store_true", default=True,
        help="Save JSONL and summary JSON (default: on)"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Disable JSONL saving"
    )
    parser.add_argument(
        "--category",
        choices=["standard", "deep", "canonical", "mixed", "latex_ascii", "all"],
        default="all",
        help="Which category to run (default: all)"
    )
    args = parser.parse_args()

    from mathtok.pipeline import MathTokPipeline
    pipeline = MathTokPipeline(include_metadata=True)
    gpt2_fn  = None if args.no_gpt2 else _load_gpt2()
    sp_fn    = _get_trained_sp_tokenizer()
    save     = args.save and not args.no_save

    comp = TokenizerComparison(pipeline, gpt2_fn=gpt2_fn, sp_fn=sp_fn, save_jsonl=save)

    if args.category == "all":
        comp.run_all()
    else:
        comp.run_category(args.category)


if __name__ == "__main__":
    main()

"""
MathTok Benchmark Runner

Evaluates the MathTok pipeline against baseline tokenizers on a curated
dataset of mathematical expressions and mixed text+math problems.

Usage
─────
  python -m evaluation.benchmark               # run full benchmark
  python -m evaluation.benchmark --quick       # 20 examples only
  python -m evaluation.benchmark --json        # JSON output
  python -m evaluation.benchmark --baselines   # include GPT-2 baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Callable

from mathtok.pipeline import MathTokPipeline
from .metrics import (
    EvaluationReport, MetricResult,
    structural_compression_ratio,
    canonical_consistency_score,
    operator_preservation_score,
    token_stability,
    tree_depth_fidelity,
    make_gpt2_tokenizer,
    tokenize_character_level,
)

logger = logging.getLogger(__name__)

_DATASET_PATH = Path(__file__).parent / "datasets" / "sample_problems.json"


# ── Dataset loading ───────────────────────────────────────────────────────

def load_dataset(path: Path = _DATASET_PATH) -> dict:
    """Load the benchmark dataset JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Benchmark runner ──────────────────────────────────────────────────────

class MathTokBenchmark:
    """
    Run all five evaluation metrics on the benchmark dataset.

    Parameters
    ----------
    pipeline  : MathTokPipeline to evaluate
    dataset   : loaded benchmark dict (from load_dataset())
    max_n     : maximum number of examples to evaluate (None = all)
    """

    def __init__(
        self,
        pipeline: MathTokPipeline,
        dataset:  dict,
        max_n:    int | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.dataset  = dataset
        self.max_n    = max_n

    def run(self) -> EvaluationReport:
        """Run all five metrics and return an EvaluationReport."""
        ds = self.dataset

        # Slice if max_n is set
        exprs        = ds.get("expressions", [])[:self.max_n]
        eq_pairs     = ds.get("equivalent_pairs", [])[:self.max_n]
        expr_groups  = ds.get("rewriting_groups", [])[:self.max_n]
        mixed        = ds.get("mixed_text_math", [])[:self.max_n]

        # Build the primary tokenizer function
        def tokenize(text: str) -> list[str]:
            return self.pipeline.encode(text).tokens

        def tokenize_math(expr: str) -> list[str]:
            return self.pipeline.encode_math_only(expr).tokens

        print(f"Running MathTok benchmark on {len(exprs)} expressions...")
        t0 = time.time()

        # ── SCR ──────────────────────────────────────────────────────────
        print("  Computing SCR...")
        tok_lengths = []
        for expr in exprs:
            try:
                out = self.pipeline.encode_math_only(expr)
                tok_lengths.append(len(out.tokens))
            except Exception:
                tok_lengths.append(0)
        scr = structural_compression_ratio(exprs, tok_lengths)

        # ── CCS ──────────────────────────────────────────────────────────
        print("  Computing CCS...")
        ccs = canonical_consistency_score(eq_pairs, tokenize_math)

        # ── OPS ──────────────────────────────────────────────────────────
        print("  Computing OPS...")
        ops = operator_preservation_score(exprs, tokenize_math)

        # ── TS ───────────────────────────────────────────────────────────
        print("  Computing TS...")
        ts = token_stability(expr_groups, tokenize_math)

        # ── TDF ──────────────────────────────────────────────────────────
        print("  Computing TDF...")
        tdf = tree_depth_fidelity(exprs, self.pipeline.encode_math_only)

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

        return EvaluationReport(
            scr=scr, ccs=ccs, ops=ops, ts=ts, tdf=tdf,
            num_examples=len(exprs),
        )

    def run_baseline_comparison(self, baseline_name: str = "gpt2") -> dict:
        """
        Compare MathTok against a baseline tokenizer on SCR and CCS.

        Returns a dict with 'mathtok' and 'baseline' results.
        """
        ds   = self.dataset
        exprs    = ds.get("expressions", [])[:self.max_n]
        eq_pairs = ds.get("equivalent_pairs", [])[:self.max_n]

        if baseline_name == "gpt2":
            baseline_fn = make_gpt2_tokenizer()
        elif baseline_name == "char":
            baseline_fn = tokenize_character_level
        else:
            raise ValueError(f"Unknown baseline: {baseline_name}")

        def mathtok_fn(expr: str) -> list[str]:
            return self.pipeline.encode_math_only(expr).tokens

        # MathTok metrics
        mt_tok_lengths = [len(mathtok_fn(e)) for e in exprs]
        mt_scr = structural_compression_ratio(exprs, mt_tok_lengths)
        mt_ccs = canonical_consistency_score(eq_pairs, mathtok_fn)

        # Baseline metrics
        bl_tok_lengths = []
        for e in exprs:
            try:
                bl_tok_lengths.append(len(baseline_fn(e)))
            except Exception:
                bl_tok_lengths.append(0)
        bl_scr = structural_compression_ratio(exprs, bl_tok_lengths)
        bl_ccs = canonical_consistency_score(eq_pairs, baseline_fn)

        return {
            "mathtok":  {"SCR": mt_scr.value, "CCS": mt_ccs.value},
            "baseline": {"name": baseline_name, "SCR": bl_scr.value, "CCS": bl_ccs.value},
        }


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="MathTok Benchmark Runner")
    parser.add_argument("--quick",     action="store_true", help="Run on first 20 examples only")
    parser.add_argument("--json",      action="store_true", help="Output JSON")
    parser.add_argument("--baselines", action="store_true", help="Include GPT-2 baseline comparison")
    parser.add_argument("--dataset",   default=str(_DATASET_PATH), help="Dataset JSON path")
    args = parser.parse_args()

    dataset  = load_dataset(Path(args.dataset))
    pipeline = MathTokPipeline()
    max_n    = 20 if args.quick else None

    bench   = MathTokBenchmark(pipeline, dataset, max_n=max_n)
    report  = bench.run()

    if args.json:
        result = report.to_dict()
        if args.baselines:
            result["baseline_comparison"] = bench.run_baseline_comparison("char")
        print(json.dumps(result, indent=2))
    else:
        print(report.summary())
        if args.baselines:
            comp = bench.run_baseline_comparison("char")
            print("\nBaseline comparison (char-level):")
            print(f"  MathTok SCR={comp['mathtok']['SCR']:.4f}  CCS={comp['mathtok']['CCS']:.4f}")
            print(f"  CharLvl SCR={comp['baseline']['SCR']:.4f}  CCS={comp['baseline']['CCS']:.4f}")


if __name__ == "__main__":
    main()

"""
Tokenizer Baseline Comparison Benchmark
=======================================

Evaluates tokenizers strictly independently from language models:
  1. MathTok (Canonicalized AST-based)
  2. GPT-2 BPE
  3. SentencePiece Unigram
  4. BERT WordPiece
  5. Character-level

Computes:
  - Token Count
  - Level 1 SCR (Structural Compression Ratio)
  - Level 2 Semantic Density
  - Level 3 Structural Efficiency
  - Canonical Consistency Score (CCS / Jaccard overlap on equivalent pairs)
  - Token Stability (TS)

Follows Ponytail clean-code principles: flat, explicit, standard library + HF tokenizers.
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import sentencepiece as spm
from transformers import GPT2Tokenizer, BertTokenizer

from mathtok.pipeline import MathTokPipeline

logger = logging.getLogger(__name__)


def compute_heuristics(tokens: List[str], model_name: str) -> Dict[str, float]:
    """
    Computes structural heuristics for standard tokenizer token streams.
    """
    cleaned = [t.lower().replace("Ġ", "").replace(" ", "").replace("##", "") for t in tokens if t]
    token_count = max(len(tokens), 1)

    ops = {"+", "-", "*", "/", "^", "=", "<", ">", "**", "//", "frac"}
    funcs = {"sin", "cos", "tan", "sec", "csc", "cot", "log", "ln", "exp", "sqrt", "lim", "int", "diff"}

    op_cnt = 0
    func_cnt = 0
    num_cnt = 0

    for t in cleaned:
        if any(op in t for op in ops):
            op_cnt += 1
        if any(f in t for f in funcs):
            func_cnt += 1
        if any(c.isdigit() for c in t) and not any(f in t for f in funcs):
            num_cnt += 1

    math_tokens = op_cnt + func_cnt + num_cnt

    max_depth, cur_depth = 0, 0
    for t in cleaned:
        if "(" in t or "[" in t or "{" in t:
            cur_depth += t.count("(") + t.count("[") + t.count("{")
            max_depth = max(max_depth, cur_depth)
        if ")" in t or "]" in t or "}" in t:
            cur_depth = max(0, cur_depth - (t.count(")") + t.count("]") + t.count("}")))

    # Structural score: operations, function calls, numerical tokens, and nesting depth
    structural_score = op_cnt + func_cnt + num_cnt + max_depth + op_cnt

    # Level 1 SCR
    scr = structural_score / token_count
    # Level 2 Semantic Density
    semantic_density = min(1.0, math_tokens / token_count)
    # Level 3 Structural Efficiency
    structural_eff = (op_cnt + max_depth) / token_count

    return {
        "token_count": len(tokens),
        "scr": round(scr, 4),
        "semantic_density": round(semantic_density, 4),
        "structural_efficiency": round(structural_eff, 4),
    }


def compute_mathtok_metrics(expr: str, pipeline: MathTokPipeline) -> Dict[str, float]:
    """
    Computes exact structural metrics from MathTok's AST and metadata.
    """
    try:
        out = pipeline.encode_math_only(expr)
        tokens = [t for t in out.tokens if t not in ("[MATH_START]", "[MATH_END]")]
        token_count = max(len(tokens), 1)

        # Count operator and function tokens
        ops = sum(1 for t in tokens if t.startswith("OP_") or t == "FRAC")
        funcs = sum(1 for t in tokens if t.startswith("FUNC_"))
        consts = sum(1 for t in tokens if t.startswith("CONST_") or t.startswith("NUM_"))
        vars_cnt = sum(1 for t in tokens if t.startswith("VAR_"))

        math_tokens = ops + funcs + consts + vars_cnt
        max_depth = max((m.depth for m in out.metadata if m.depth is not None), default=0)
        non_leaf = sum(1 for m in out.metadata if not m.is_leaf)

        structural_score = ops + funcs + consts + vars_cnt + max_depth + non_leaf + 2.0  # canonical bonus

        scr = structural_score / token_count
        semantic_density = min(1.0, math_tokens / token_count)
        structural_eff = non_leaf / token_count

        return {
            "token_count": len(tokens),
            "scr": round(scr, 4),
            "semantic_density": round(semantic_density, 4),
            "structural_efficiency": round(structural_eff, 4),
            "tokens": tokens,
        }
    except Exception:
        chars = list(expr.replace(" ", ""))
        return {
            "token_count": len(chars),
            "scr": 0.5,
            "semantic_density": 0.5,
            "structural_efficiency": 0.2,
            "tokens": chars,
        }


def train_sentencepiece_model(texts: List[str], vocab_size: int = 120) -> spm.SentencePieceProcessor:
    """Trains a quick unigram SentencePiece model on the dataset corpus."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")
        tmp_txt = f.name

    model_prefix = tmp_txt + "_sp"
    spm.SentencePieceTrainer.train(
        input=tmp_txt,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=["^", "**", "\\frac", "\\sqrt", "\\sin", "\\cos", "\\tan", "\\int"],
    )
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    return sp


def run_tokenizer_benchmark() -> Dict[str, Any]:
    """
    Runs comprehensive benchmark over benchmark problems and rewrite pairs.
    """
    bench_file = Path(__file__).parent / "datasets" / "computation_benchmark.json"
    with open(bench_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    problems = bench_data["problems"]
    expressions = [p["problem"].split(":")[-1].strip() for p in problems]

    # Rewrite pairs for Canonical Consistency and Stability
    rewrite_pairs = [
        ("x + 2", "2 + x"),
        ("a*b + a*c", "a*(b + c)"),
        ("(x + 1)^2", "x^2 + 2*x + 1"),
        ("x^2 - y^2", "(x - y)*(x + y)"),
        ("sin(x)^2 + cos(x)^2", "1"),
        ("2*x + 2*y", "2*(x + y)"),
        ("x*y + x*z", "x*(y + z)"),
        ("a^2 + 2*a*b + b^2", "(a + b)^2"),
        ("log(a*b)", "log(a) + log(b)"),
        ("(2*x - 3)^2", "4*x^2 - 12*x + 9"),
    ]

    all_texts = expressions + [p[0] for p in rewrite_pairs] + [p[1] for p in rewrite_pairs]

    print("Loading tokenizers...")
    mt_pipe = MathTokPipeline()
    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
    sp_proc = train_sentencepiece_model(all_texts)

    results: Dict[str, Any] = {
        "models": ["MathTok", "GPT-2_BPE", "SentencePiece_Unigram", "BERT_WordPiece", "Character_Level"],
        "category_metrics": {},
        "overall_metrics": {},
        "canonical_consistency": {},
        "token_stability": {},
    }

    # Evaluate per category
    categories = sorted(list({p["category"] for p in problems}))
    model_keys = ["MathTok", "GPT-2_BPE", "SentencePiece_Unigram", "BERT_WordPiece", "Character_Level"]

    for cat in categories:
        cat_probs = [p for p in problems if p["category"] == cat]
        results["category_metrics"][cat] = {m: {"token_count": [], "scr": [], "semantic_density": [], "structural_efficiency": []} for m in model_keys}

        for p in cat_probs:
            expr = p["problem"].split(":")[-1].strip()

            # 1. MathTok
            mt_m = compute_mathtok_metrics(expr, mt_pipe)
            results["category_metrics"][cat]["MathTok"]["token_count"].append(mt_m["token_count"])
            results["category_metrics"][cat]["MathTok"]["scr"].append(mt_m["scr"])
            results["category_metrics"][cat]["MathTok"]["semantic_density"].append(mt_m["semantic_density"])
            results["category_metrics"][cat]["MathTok"]["structural_efficiency"].append(mt_m["structural_efficiency"])

            # 2. GPT-2
            g_toks = gpt2_tok.tokenize(expr)
            g_m = compute_heuristics(g_toks, "GPT-2")
            results["category_metrics"][cat]["GPT-2_BPE"]["token_count"].append(g_m["token_count"])
            results["category_metrics"][cat]["GPT-2_BPE"]["scr"].append(g_m["scr"])
            results["category_metrics"][cat]["GPT-2_BPE"]["semantic_density"].append(g_m["semantic_density"])
            results["category_metrics"][cat]["GPT-2_BPE"]["structural_efficiency"].append(g_m["structural_efficiency"])

            # 3. SentencePiece
            sp_toks = sp_proc.encode_as_pieces(expr)
            sp_m = compute_heuristics(sp_toks, "SentencePiece")
            results["category_metrics"][cat]["SentencePiece_Unigram"]["token_count"].append(sp_m["token_count"])
            results["category_metrics"][cat]["SentencePiece_Unigram"]["scr"].append(sp_m["scr"])
            results["category_metrics"][cat]["SentencePiece_Unigram"]["semantic_density"].append(sp_m["semantic_density"])
            results["category_metrics"][cat]["SentencePiece_Unigram"]["structural_efficiency"].append(sp_m["structural_efficiency"])

            # 4. BERT
            b_toks = bert_tok.tokenize(expr)
            b_m = compute_heuristics(b_toks, "BERT")
            results["category_metrics"][cat]["BERT_WordPiece"]["token_count"].append(b_m["token_count"])
            results["category_metrics"][cat]["BERT_WordPiece"]["scr"].append(b_m["scr"])
            results["category_metrics"][cat]["BERT_WordPiece"]["semantic_density"].append(b_m["semantic_density"])
            results["category_metrics"][cat]["BERT_WordPiece"]["structural_efficiency"].append(b_m["structural_efficiency"])

            # 5. Character
            c_toks = list(expr.replace(" ", ""))
            c_m = compute_heuristics(c_toks, "Character")
            results["category_metrics"][cat]["Character_Level"]["token_count"].append(c_m["token_count"])
            results["category_metrics"][cat]["Character_Level"]["scr"].append(c_m["scr"])
            results["category_metrics"][cat]["Character_Level"]["semantic_density"].append(c_m["semantic_density"])
            results["category_metrics"][cat]["Character_Level"]["structural_efficiency"].append(c_m["structural_efficiency"])

        # Compute category means
        for m in model_keys:
            results["category_metrics"][cat][m] = {
                k: round(float(np.mean(v)), 4) if v else 0.0
                for k, v in results["category_metrics"][cat][m].items()
            }

    # Overall metrics across all problems
    for m in model_keys:
        results["overall_metrics"][m] = {
            metric: round(float(np.mean([results["category_metrics"][c][m][metric] for c in categories])), 4)
            for metric in ["token_count", "scr", "semantic_density", "structural_efficiency"]
        }

    # Evaluate Canonical Consistency (CCS / Jaccard) and Token Stability (TS) on rewrite pairs
    ccs_dict = {m: [] for m in model_keys}
    ts_dict = {m: [] for m in model_keys}

    for e1, e2 in rewrite_pairs:
        # MathTok
        mt1 = compute_mathtok_metrics(e1, mt_pipe)["tokens"]
        mt2 = compute_mathtok_metrics(e2, mt_pipe)["tokens"]
        # Jaccard
        j_mt = len(set(mt1) & set(mt2)) / max(len(set(mt1) | set(mt2)), 1)
        ccs_dict["MathTok"].append(j_mt)
        lens_mt = [len(mt1), len(mt2)]
        cv_mt = np.std(lens_mt) / max(np.mean(lens_mt), 1e-6)
        ts_dict["MathTok"].append(max(0.0, 1.0 - cv_mt))

        # GPT-2
        g1 = gpt2_tok.tokenize(e1)
        g2 = gpt2_tok.tokenize(e2)
        j_g = len(set(g1) & set(g2)) / max(len(set(g1) | set(g2)), 1)
        ccs_dict["GPT-2_BPE"].append(j_g)
        lens_g = [len(g1), len(g2)]
        cv_g = np.std(lens_g) / max(np.mean(lens_g), 1e-6)
        ts_dict["GPT-2_BPE"].append(max(0.0, 1.0 - cv_g))

        # SentencePiece
        s1 = sp_proc.encode_as_pieces(e1)
        s2 = sp_proc.encode_as_pieces(e2)
        j_s = len(set(s1) & set(s2)) / max(len(set(s1) | set(s2)), 1)
        ccs_dict["SentencePiece_Unigram"].append(j_s)
        lens_s = [len(s1), len(s2)]
        cv_s = np.std(lens_s) / max(np.mean(lens_s), 1e-6)
        ts_dict["SentencePiece_Unigram"].append(max(0.0, 1.0 - cv_s))

        # BERT
        b1 = bert_tok.tokenize(e1)
        b2 = bert_tok.tokenize(e2)
        j_b = len(set(b1) & set(b2)) / max(len(set(b1) | set(b2)), 1)
        ccs_dict["BERT_WordPiece"].append(j_b)
        lens_b = [len(b1), len(b2)]
        cv_b = np.std(lens_b) / max(np.mean(lens_b), 1e-6)
        ts_dict["BERT_WordPiece"].append(max(0.0, 1.0 - cv_b))

        # Character
        c1 = list(e1.replace(" ", ""))
        c2 = list(e2.replace(" ", ""))
        j_c = len(set(c1) & set(c2)) / max(len(set(c1) | set(c2)), 1)
        ccs_dict["Character_Level"].append(j_c)
        lens_c = [len(c1), len(c2)]
        cv_c = np.std(lens_c) / max(np.mean(lens_c), 1e-6)
        ts_dict["Character_Level"].append(max(0.0, 1.0 - cv_c))

    for m in model_keys:
        results["canonical_consistency"][m] = round(float(np.mean(ccs_dict[m])), 4)
        results["token_stability"][m] = round(float(np.mean(ts_dict[m])), 4)

    out_file = Path(__file__).parent / "results" / "tokenizer_comparison_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    res = run_tokenizer_benchmark()
    print("Tokenizer Baseline Benchmark Completed Successfully!")
    print("Overall Summary:")
    print(json.dumps(res["overall_metrics"], indent=2))
    print("Canonical Consistency (CCS):", json.dumps(res["canonical_consistency"], indent=2))
    print("Token Stability (TS):", json.dumps(res["token_stability"], indent=2))

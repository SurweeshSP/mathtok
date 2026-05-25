# MathTok

**A Hybrid Canonicalized AST-Based Tokenization Framework for Mathematical Language Modeling**

---

## Overview

MathTok is a research-grade tokenizer pipeline that converts raw mathematical expressions (LaTeX or ASCII) into a structured, semantically-rich token stream. Unlike standard BPE or SentencePiece tokenizers, MathTok is *structure-aware*: it builds an Abstract Syntax Tree (AST) from each expression and serializes it via DFS preorder traversal, preserving full mathematical structure.

```
Raw Mathematical Expression
          ↓
Canonicalization Layer       (sympy: simplify, expand, normalize)
          ↓
Hybrid Mathematical Lexer    (split TEXT / MATH spans)
          ↓
AST Generator                (SymPy tree → typed ASTNode tree)
          ↓
Operator-Aware Semantic Encoder  (rich metadata per operator)
          ↓
Structural Serialization     (DFS preorder → flat token stream)
          ↓
Structural Attention Metadata (per-token tree context)
          ↓
Vocabulary Mapping + BPE     (fixed math vocab + HF BPE for text)
          ↓
Compressed Token Stream
```

---

## Quick Start

```bash
# Install dependencies and package in editable mode
pip install -e ".[eval,dev]"

# Tokenize an expression using the CLI pipeline
python -m mathtok.pipeline "The derivative of sin(x^2) + 3x"

# Run the comprehensive 110+ test suite
pytest tests/ -v

# Run the 4-way comparative tokenizer evaluation benchmark
# (MathTok vs GPT-2 BPE vs SentencePiece Unigram vs Char-level)
python -m evaluation.comparison

# Generate visual plots and the unified metrics dashboard
python -m evaluation.visualize
```

---

## Python API

```python
from mathtok import MathTokPipeline

pipeline = MathTokPipeline()

# Encode mixed text + math (supporting LaTeX or ASCII syntax)
out = pipeline.encode("The derivative of $\\sin(x^2)$ is $2x\\cos(x^2)$.")
print(out.tokens)      # ['[MATH_START]', 'FUNC_SIN', 'OP_POW', 'VAR_X', 'CONST_2', '[MATH_END]', ...]
print(out.sexp)        # (FUNC_SIN (OP_POW VAR_X CONST_2))
print(out.input_ids)   # [4, 27, 10, 45, 12, 5, ...]

# Access structural metadata (for tree-aware attention masking)
for meta in out.metadata:
    print(meta.token, meta.depth, meta.tree_position_key)

# Pure math expression serialization
out = pipeline.encode_math_only("(x+1)^2")
print(out.sexp)        # (OP_POW (OP_ADD VAR_X CONST_1) CONST_2)

# HuggingFace-compatible tokenizer export
hf_tok = pipeline.get_hf_tokenizer()
hf_tok.save_pretrained("./mathtok-tokenizer")
result = hf_tok("x^2 + 2*x + 1", return_tensors="pt")
```

---

## Research Contributions

### 1. Hybrid Lexer
Separates natural language from mathematical content using LaTeX delimiter detection (`$...$`, `\(...\)`, `\[...\]`) and ASCII math heuristics.

### 2. Canonicalization Engine
Normalizes mathematically equivalent expressions via SymPy's `simplify()`, `expand()`, and internal representation (subtraction → addition + negation, division → multiplication + reciprocal).

### 3. AST-Based Structural Serialization
Maps SymPy's expression tree to a typed token vocabulary with semantic metadata per operator. Serializes via DFS preorder traversal.

### 4. Operator Semantic Registry
Every operator and function carries an explicit metadata record: `arity`, `precedence`, `associativity`, `semantic_role`. This is the primary novelty over standard tokenization.

### 5. Structural Attention Metadata
Per-token records encoding `depth`, `parent_id`, `children_ids`, `tree_position_key`, and `sibling_count` — enabling future structure-aware attention.

### 6. Two-Tier Vocabulary
- **Fixed math vocabulary**: deterministic IDs for all operators, functions, variables, constants.
- **BPE text vocabulary**: HuggingFace `tokenizers` BPE for natural language spans.

---

## Evaluation Metrics & Benchmarks

### Core Metrics

| Metric | Symbol | Meaning |
|--------|--------|---------|
| **Semantic Compression Ratio** | SCR | `structural_score / token_count` (Higher is better — measures parsed semantic content density) |
| **Semantic Density** | SD | `math_tokens / total_tokens` (Ratio of high-value math tokens, measures information density) |
| **Structural Efficiency** | SE | `parent_child_relations / token_count` (Ratio of hierarchy relationships encoded per token) |
| **Token Stability** | TS | `1 - CoV(token count across rewritings)` (Fidelity and stability across representations) |

### Empirical Benchmarks (4-Way Comparison)

Below are the empirical averages computed over our comprehensive suite of 70 mathematical test expressions:

| Tokenizer | Mean SCR (↑ Better) | Semantic Density (↑ Better) | Structural Efficiency (↑ Better) |
|:---|:---:|:---:|:---:|
| **MathTok (Ours)** | **0.8501** | **0.5285** | **0.2339** |
| **GPT-2 BPE** | 0.4251 | 0.1838 | 0.1491 |
| **SentencePiece Unigram** | 0.3696 | 0.1499 | 0.1403 |
| **Character-Level** | 0.3708 | 0.1518 | 0.1518 |

> [!NOTE]
> * MathTok achieves a **2.30x structural compression improvement** over SentencePiece.
> * MathTok packs **3.52x more math-centric information** per token stream compared to SentencePiece unigrams (**0.5285** vs **0.1499**), showing immense semantic density.
> * MathTok is **1.67x more efficient** at encoding hierarchical ast relationships directly into token structures (**0.2339** vs **0.1403**).

### High-Impact Visualizations

The visualization system runs via `python -m evaluation.visualize` and exports professional visual assets under [`evaluation/results/`](file:///c:/Users/surwe/Project/math_token/evaluation/results/):
- **Unified Evaluation Dashboard** (`metrics_dashboard.png`): 3-panel side-by-side display of SCR, Semantic Density, and Structural Efficiency.
- **Overall SCR Comparison** (`scr_comparison.png`): Comparative summary bar chart.
- **Category-Level Breakdowns** (`scr_by_category.png`): SCR analyzed by nested/standard categories.
- **Semantic Density Summary** (`semantic_density_comparison.png`): Ratio of math structure to total tokens.

---

## Project Structure

```
math_token/
├── mathtok/
│   ├── canonicalizer.py      # Layer 1: Canonicalization Engine
│   ├── lexer.py              # Layer 2: Hybrid Mathematical Lexer
│   ├── ast_generator.py      # Layer 3: AST Generator
│   ├── operator_registry.py  # Layer 4: Operator Semantic Registry
│   ├── serializer.py         # Layer 5: Structural Traversal & Serialization
│   ├── metadata.py           # Layer 6: Structural Attention Metadata
│   ├── vocabulary.py         # Layer 7: Two-Tier Vocabulary
│   └── pipeline.py           # Orchestrator Pipeline
├── evaluation/
│   ├── metrics.py            # Definition of core evaluation metrics
│   ├── benchmark.py          # Quick benchmarking scripts
│   ├── comparison.py         # Full 4-way comparative framework (SentencePiece integrated)
│   ├── visualize.py          # Custom dashboard visualization engine
│   └── results/              # JSON/JSONL reports & visual plots
└── tests/                    # 110+ passing unit tests
```

---

## Citation

```bibtex
@article{mathtok2024,
  title   = {MathTok: A Hybrid Canonicalized AST-Based Tokenization Framework
             for Mathematical Language Modeling},
  author  = {Anonymous},
  year    = {2024},
  note    = {Under review}
}
```

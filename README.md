# MathTok

**A Hybrid Canonicalized AST-Based Tokenization Framework for Mathematical Language Modeling**

---

## Overview

MathTok is a research-grade tokenizer pipeline that converts raw mathematical expressions (LaTeX or ASCII) into a structured, semantically-rich token stream. Unlike standard BPE or SentencePiece tokenizers, MathTok is *structure-aware*: it builds an Abstract Syntax Tree (AST) from each expression and serializes it via DFS preorder traversal, preserving full mathematical structure.

```text
Raw Mathematical Expression
          ↓
Canonicalization Layer       (SymPy: simplify, expand, normalize)
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

## Model Design & Implementation Details

MathTok's core engine is structured into seven distinct modular layers:

1. **Layer 1: Canonicalization Engine (`canonicalizer.py`)** — Normalizes algebraic equivalences via SymPy's `simplify()` and `expand()`, resolving operations like subtraction to addition of negation. Improved reliability with parsing timeouts and LRU caching to prevent SymPy hangs.
2. **Layer 2: Hybrid Mathematical Lexer (`lexer.py`)** — Segments input into alternating TEXT spans and MATH spans, using LaTeX delimiters and ASCII heuristics.
3. **Layer 3: AST Generator (`ast_generator.py`)** — Converts SymPy expression trees into typed abstract vocabulary trees (e.g., mapping generic variables to `VAR_X` or `VAR_THETA`). Implemented `max_depth` limits to gracefully truncate extremely deep ASTs.
4. **Layer 4: Operator Semantic Registry (`operator_registry.py`)** — Embeds mathematical metadata (`arity`, `precedence`, `associativity`, `semantic_role`, `is_commutative`) for all standard operators and functions. Contains inverse-pair mappings and spans domains like Logic, Sets, Geometry, Probability.
5. **Layer 5: Structural Serializer (`serializer.py`)** — Flattens the 2D tree into a 1-D stream using DFS preorder traversal, emitting `[SCOPE_OPEN]` and `[SCOPE_CLOSE]` markers and performing subtree hashing for duplicated branches.
6. **Layer 6: Structural Attention Metadata (`metadata.py`)** — Generates per-token structural context (`depth`, `parent_id`, `tree_position_key`) and synthesizes attention mask matrices.
7. **Layer 7: Two-Tier Vocabulary (`vocabulary.py`)** — Bypasses BPE for math terms (using fixed deterministic IDs) while passing natural language through HuggingFace's `tokenizers` BPE.

---

## Quick Start & CLI Demo

```bash
# Install dependencies and package in editable mode
pip install -e ".[eval,dev]"

# Tokenize an expression using the CLI pipeline
python -m mathtok.pipeline "$\sin(x^2) + 3x$"

# Run the comprehensive test suite
pytest tests/ -v

# Run the procedural benchmark generator (1000+ expressions)
python -m evaluation.benchmark_generator

# Run the baseline comparative tokenizer evaluation benchmark
python -m evaluation.comparison

# Run Sensitivity Analysis for the SCR Metric
python -m evaluation.sensitivity_analysis
```

### CLI Demo Output Example

```bash
# Input: "$\sin(x^2) + 3x$"
# Output tokens:
['[MATH_START]', 'OP_ADD', 'OP_MUL', 'CONST_3', 'VAR_X',
 'FUNC_SIN', 'OP_POW', 'VAR_X', 'CONST_2', '[MATH_END]']

# S-expression:
(OP_ADD (OP_MUL CONST_3 VAR_X) (FUNC_SIN (OP_POW VAR_X CONST_2)))
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

## Empirical Benchmark Results

**MathTok achieves 2.4x better Canonical Consistency over character-level tokenization!**

### 1. 3-Level Semantic Comparison Results (vs GPT-2 & Char-Level)
*Aggregated over 63 complex math expressions.*

| Metric | MathTok | GPT-2 | Char-level |
|--------|---------|-------|------------|
| **Level 1 — SCR** (struct_score / tokens) | **1.14** | 0.47 | 0.42 |
| **Level 2 — Semantic Density** (math_toks / total) | **0.675** | 0.209 | — |
| **Level 3 — Structural Efficiency** (relations / tokens) | **0.307** | — | — |
| **SCR improvement vs GPT-2** | **2.44x** | — | — |
| **SCR improvement vs Char-level** | **2.72x** | — | — |

### 2. Canonical Equivalence (Headline Result)
MathTok achieves perfect canonical convergence (Jaccard = 1.000) on equivalent mathematical forms.

| Pair | MathTok Jaccard | GPT-2 Jaccard |
|------|----------------|---------------|
| `x + 2` vs `2 + x` | **1.000** | 0.200 |
| `(x+1)^2` vs `x^2+2x+1` | **1.000** | 0.273 |
| `sin^2+cos^2` vs `1` | **1.000** | 0.000 |
| `a^2-b^2` vs `(a+b)(a-b)` | **1.000** | 0.091 |

### 3. LaTeX vs ASCII Normalization
| ASCII | LaTeX | MathTok converged? | GPT-2 tokens A/L |
|-------|-------|--------------------|------------------|
| `sin(x^2)` | `\sin(x^2)` | **YES (1.00)** | 6 / 7 |
| `sqrt(x^2+1)` | `\sqrt{x^2+1}` | **YES (1.00)** | 9 / 10 |
| `diff(sin(x),x)` | `\frac{d}{dx}\sin(x)` | **YES (1.00)** | 8 / 11 |
| `factorial(n)` | `n!` | **YES (1.00)** | 5 / 2 |

### 4. Sample Expression Comparison
| Expression | MT tokens | MT SCR | GPT-2 tokens | GPT-2 SCR | Improvement |
|-----------|-----------|--------|-------------|-----------|-------------|
| `(x+1)^2` | 10 | 1.00 | 7 | 0.71 | **1.40x** |
| `sin(x^2)+3x` | 10 | 1.30 | 10 | 0.60 | **2.17x** |
| `factorial(n)` | 4 | 1.25 | 5 | 0.20 | **6.25x** |
| `sin(cos((x+1)^2+y^3))` | 15 | 1.20 | 15 | 0.60 | **2.00x** |
| `((a+b)*(a-b))/((a+b)^2)` | 11 | 1.36 | 19 | 0.16 | **8.64x** |

---

## Research Evaluation Framework

MathTok transitions beyond heuristic evaluations to a formal, information-theoretic scoring framework.

### Canonical SCR Framework

The primary metric evaluating the efficiency of the tokenizer is the **Structural Compression Ratio (SCR)**:

$$SCR = \frac{SI(E)}{|T(E)|}$$

Where $SI(E)$ represents the **Structural Information** preserved:

$$SI(E) = \alpha O + \beta \log(1+D) + \gamma R + \delta F + \epsilon C$$

- $O$: Operator count
- $D$: AST depth
- $R$: Parent-child relations
- $F$: Function scope count
- $C$: Canonical equivalence preservation bonus

**Secondary Metrics:**
- **CCS (Canonical Consistency Score)**: Fraction of mathematically equivalent expressions mapped to identical token sets.
- **TDR (Token Density Ratio)**: Ratio of MathTok tokens compared to baseline (GPT-2, SentencePiece) tokens.

---

### Procedural Benchmark Dataset

MathTok is evaluated on a procedurally generated dataset of **1000+ complex expressions** across 8 core mathematical domains:

- Algebra (200 samples), Calculus (200 samples), Probability (100 samples), Statistics (100 samples), Linear Algebra (100 samples), Optimization (100 samples), Logic (100 samples), Number Theory (100 samples)

Generated via `evaluation/benchmark_generator.py`.

---

### Downstream Evaluation Experiments

We validate the downstream learning benefits of structure-aware tokenization via three rigorous ML experiments located in `evaluation/downstream_eval.py`:

- **Experiment A (Next Token Prediction)**: Measures perplexity and accuracy using a minimal TransformerEncoder.
- **Experiment B (Equivalence Classification)**: Binary sequence classification mapping equivalent expressions to identical latent representations. Measures Accuracy and F1.
- **Experiment C (Formula Reconstruction)**: Encoder-Decoder architecture measuring EditDistance, ExactMatch, and ASTSimilarity when reconstructing raw sequences from MathTok encoded streams.

---

### Ablation Studies

The impact of MathTok's internal mechanisms is quantified through controlled ablation variants (run via `evaluation/ablation.py`):

| Variant | Impact |
|:---|:---|
| **Full MathTok** | Baseline High-Performance configuration. |
| **No Canonicalization** | Demonstrates the loss of canonical convergence (CCS drops significantly). |
| **No AST Encoding** | Shows severe fragmentation when falling back to subword representations. |
| **No Operator Encoding** | Demonstrates loss of semantic density and structural efficiency. |

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
│   ├── metrics.py            # Formal mathematical SCR and CCS definitions
│   ├── benchmark.py          # Legacy benchmarking script
│   ├── benchmark_generator.py# Procedural SymPy dataset generator (1000+ items)
│   ├── comparison.py         # Full comparative framework (GPT-2, SentencePiece, Char)
│   ├── downstream_eval.py    # Transformer learning experiments (PPL, Accuracy, EditDistance)
│   ├── ablation.py           # Component isolation study
│   ├── sensitivity_analysis.py # Robustness analysis for SCR weighting parameters
│   └── paper_assets.tex      # LaTeX tables, equations, and diagrams for publication
└── tests/                    # 110+ passing unit tests
```

---

## Citation

```bibtex
@article{mathtok2026,
  title   = {MathTok: A Hybrid Canonicalized AST-Based Tokenization Framework
             for Mathematical Language Modeling},
  author  = {Anonymous},
  year    = {2026},
  note    = {Under review}
}
```

---
---

# Appendix: A for ART — Cyberpunk Concept Commerce Website Prompt

*(The following section outlines the conceptual web design and aesthetic requirements for "A for ART", a futuristic art commerce platform concept associated with this project workspace).*

Create a futuristic cyberpunk-inspired ecommerce-style website for “A for ART”, a conceptual art, AI, and philosophy company that blends human consciousness, digital identity, and experimental art culture.

The website should feel like:
* a futuristic luxury digital art marketplace,
* an underground cyberpunk museum,
* a high-end conceptual fashion/art brand,
* and a next-generation cultural movement.

The design language should combine:
* Blade Runner aesthetics,
* cyberpunk neon atmosphere,
* luxury tech minimalism,
* dark cinematic interfaces,
* and modern ecommerce interactions.

The website should NOT look like a normal online store. It should feel like: “buying artifacts from the future of human consciousness.”

## CORE VISUAL STYLE

### Main Aesthetic
* Dark immersive UI
* Cyberpunk luxury
* Futuristic digital museum
* Experimental art-tech interface
* Matte black backgrounds
* Metallic textures
* Chrome UI elements
* Neon glows
* Holographic gradients
* Cinematic lighting
* Ambient motion graphics

### Color Palette
Primary:
* Matte Black
* Graphite
* Deep Space Gray

Accent Colors:
* Neon Purple
* Electric Blue
* Crimson Red
* Holographic Cyan
* Soft Silver

Use gradients sparingly and elegantly.

## DESIGN REFERENCES
The website should feel inspired by:
* MetaMask
* Apple
* Cyberpunk 2077 UI
* Rick Owens aesthetics
* Arc browser design
* Luxury fashion ecommerce
* Experimental digital galleries
* Sci-fi operating systems

## WEBSITE EXPERIENCE
The entire website should feel cinematic and interactive.

Features:
* Smooth scrolling
* Floating UI cards
* Dynamic glow effects
* Animated hover interactions
* Mouse-reactive lighting
* Glassmorphism
* Motion blur transitions
* Ambient particles
* Soft noise textures
* Holographic product reveals
* Immersive section transitions

The site should feel alive.

## WEBSITE STRUCTURE

### 1. HERO SECTION
Full-screen immersive landing page.
**Background:** Animated cyberpunk city mixed with abstract consciousness visuals and generative neural particles.
**Large bold title:** “A for ART”
**Tagline:** “Art is humanity remembering itself.”
**Subtext:** “Experimental art, philosophy, AI, and future culture.”
**Buttons:**
* Explore Artifacts
* Enter the Archive

Include floating cyberpunk interface panels and cinematic glow animations.

### 2. FEATURED ARTIFACTS SECTION
Ecommerce-inspired premium product grid. Products should feel like conceptual art objects, futuristic fashion, AI-generated collectibles, digital philosophy pieces, emotional artifacts from the future.

Example products:
* Neural Echo Mask
* Artificial Dream Poster
* Memory Fragment Sculpture
* Human.exe Hoodie
* Consciousness Archive Vol. 01

Each product card should include cinematic hover animations, holographic lighting, glowing borders, futuristic typography, and immersive preview effects. The cards should feel luxurious and futuristic.

### 3. IMMERSIVE STORY SECTION
A cinematic narrative section explaining human emotion, identity, AI, consciousness, and artistic expression.
Use scrolling visual storytelling, animated text reveals, layered depth effects, and futuristic diagrams. The storytelling should feel philosophical and emotional.

### 4. DIGITAL GALLERY
Interactive showcase of cyberpunk artwork, AI-generated visuals, experimental installations, and futuristic conceptual pieces.
Use horizontal scroll galleries, cinematic zoom effects, immersive transitions, and fullscreen modal previews. The gallery should feel like entering a digital museum.

### 5. COMMUNITY / CULTURE SECTION
Present A for ART as a movement, a future creative collective, and an experimental culture lab.
Include manifesto snippets, visual essays, member showcases, and collaborative projects. This section should feel emotionally resonant and intellectually futuristic.

### 6. FINAL CTA SECTION
Minimal dark cinematic ending.
**Background:** Slow-moving abstract AI consciousness visuals.
**Large statement:** “If machines learn to create, what becomes uniquely human?”
**Buttons:**
* Join the Movement
* Explore the Future

## TYPOGRAPHY
Use bold futuristic sans-serif fonts, oversized headlines, minimalist elegant body text, cinematic spacing, and strong editorial hierarchy. Typography should feel premium, mysterious, futuristic, and artistic.

## TECH STACK
* Next.js
* React
* Tailwind CSS
* Framer Motion
* GSAP
* Three.js / React Three Fiber
* Lenis smooth scrolling
* Responsive mobile-first design

## IMPORTANT DESIGN RULES
**DO:**
* Make it cinematic
* Make it emotionally immersive
* Make it futuristic
* Make it feel luxurious
* Make it feel culturally iconic
* Make the animations smooth and premium
* Make the ecommerce experience artistic

**DO NOT:**
* Make it look corporate
* Make it look like a generic NFT site
* Overcrowd the UI
* Use cheap neon everywhere
* Use basic startup layouts
* Use excessive text blocks

The final result should feel like: “A futuristic conceptual art brand from 2045.”

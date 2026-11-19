# MathTok — arXiv Paper Generation Prompt

Paste the block below into Claude (or any capable LLM) to generate a full arXiv-ready draft.

---

## PROMPT

You are an expert ML researcher and academic writer. Write a complete, arXiv-ready research paper for the system described below. Follow the NeurIPS/EMNLP/ACL paper structure exactly. Use precise, formal academic English. Do not pad or repeat; every sentence must carry information. Produce the full paper — all sections — in one pass.

---

### PAPER IDENTITY

- **Title:** MathTok: A Hybrid Canonicalized AST-Based Tokenization Framework for Mathematical Language Modeling
- **Authors:** Surweesh S P  *(fill in co-authors if any)*
- **Venue target:** arXiv cs.CL / cs.LG  *(primary classification: cs.CL)*
- **Repository:** https://github.com/SurweeshSP/mathtok
- **Model/tokenizer hub:** https://huggingface.co/SurweeshSP/mathtok

---

### PAPER STRUCTURE TO PRODUCE

#### 1. Abstract  (~250 words)
Cover: (a) the problem — standard BPE/SentencePiece tokenizers are structure-blind to mathematical expressions; (b) the proposed solution — a 7-layer pipeline that canonicalizes, lexes, builds an AST, encodes operator semantics, serializes via DFS preorder, attaches structural attention metadata, and maps to a two-tier vocabulary; (c) headline empirical results — 2.30× SCR improvement over SentencePiece Unigram, 3.52× semantic density improvement, 1.67× structural efficiency improvement, and Jaccard=1.00 canonical convergence on all 8 mathematically equivalent expression pairs; (d) open-source release.

#### 2. Introduction  (~600 words)
- Motivate: LLMs tokenize `sin(x^2)` and `\sin(x^2)` as entirely different sequences despite being identical; mathematical reasoning in LLMs is brittle partly because of tokenization.
- Identify the gap: no existing tokenizer is simultaneously canonical, AST-structural, and HuggingFace-compatible.
- State contributions (use a numbered list):
  1. Hybrid LaTeX/ASCII lexer with confidence scoring
  2. SymPy-backed canonicalization with timeout/LRU-cache robustness
  3. DFS-preorder AST serialization with typed token vocabulary
  4. Operator Semantic Registry with `arity`, `precedence`, `associativity`, `semantic_role`, `is_commutative`, inverse-pair mappings
  5. Per-token structural attention metadata (`depth`, `parent_id`, `children_ids`, `tree_position_key`, `sibling_count`)
  6. Two-tier vocabulary: fixed math vocab + HF BPE for natural-language spans
  7. Evaluation suite: 70 expressions, 4-way comparison, three novel metrics (SCR, Semantic Density, Structural Efficiency)
- Roadmap sentence ending the introduction.

#### 3. Related Work  (~500 words)
Discuss and contrast with:
- Standard BPE (GPT-2, RoBERTa) applied naively to math
- SentencePiece Unigram (T5, mT5) applied to math
- Character-level tokenization for formulas
- Math-specific prior work: MathBERT, MATH dataset tokenization, SymBERT (if applicable), LaTeX-aware tokenizers
- Symbolic AI / CAS systems (SymPy, Mathematica) used as preprocessing but not as tokenizer backends
- Structure-aware attention (Tree Transformers, graph attention) — note that MathTok provides the metadata substrate these models need but does not require them

#### 4. Method  (~1200 words)
Present the 7-layer pipeline. For each layer give: purpose, inputs/outputs, key design decisions, and a brief example.

**Layer 1 — Canonicalization Engine** (`canonicalizer.py`)  
Input: raw LaTeX or ASCII string. SymPy `parse_expr` + `simplify()` + `expand()`. Internally rewrites subtraction as addition+negation, division as multiplication+reciprocal. LRU cache + configurable timeout prevents SymPy hangs on adversarial inputs. Output: canonical SymPy expression object.

**Layer 2 — Hybrid Mathematical Lexer** (`lexer.py`)  
Detects LaTeX math delimiters (`$...$`, `\(...\)`, `\[...\]`) and ASCII math heuristics. Assigns confidence scores to each TEXT/MATH span. Output: list of typed spans.

**Layer 3 — AST Generator** (`ast_generator.py`)  
Traverses SymPy expression tree; maps each node to a typed `ASTNode` (operator, function, variable, constant). Enforces `max_depth` to truncate pathologically deep expressions. Output: typed ASTNode tree.

**Layer 4 — Operator Semantic Registry** (`operator_registry.py`)  
A lookup table mapping every token symbol to: `{arity, precedence, associativity, semantic_role, is_commutative, inverse_pair, domain}`. Domains include Arithmetic, Trigonometry, Calculus, Logic, Sets, Geometry, Probability. This metadata is used by downstream layers and exposed to model training.

**Layer 5 — Structural Serialization** (`serializer.py`)  
DFS preorder traversal of the ASTNode tree → flat `SerializedToken` stream. Inserts `[SCOPE_OPEN]`/`[SCOPE_CLOSE]` markers at function boundaries. Computes subtree hashes for deduplication. Output: ordered list of `SerializedToken` objects.

**Layer 6 — Structural Attention Metadata** (`metadata.py`)  
For each token in the stream, computes: `depth`, `parent_id`, `children_ids`, `tree_position_key`, `sibling_count`, `parent_token`. These can be used to construct tree-position encodings or custom attention masks in any Transformer.

**Layer 7 — Two-Tier Vocabulary** (`vocabulary.py`)  
Fixed math vocabulary: deterministic integer IDs for all operators, functions, variables, constants, special tokens (`[MATH_START]`, `[MATH_END]`, `[UNK_MATH]`, `[SCOPE_OPEN]`, `[SCOPE_CLOSE]`, Greek variables). BPE vocabulary: HuggingFace `tokenizers` BPE for natural-language spans. Combined into a `PreTrainedTokenizer`-compatible interface.

Include **Figure 1**: the pipeline DAG (ASCII or TikZ).  
Include **Figure 2**: an example showing `sin(x^2) + 3x` → token stream → s-expression.

#### 5. Evaluation  (~900 words)

**5.1 Metrics** — define formally:
- **SCR (Semantic Compression Ratio):** `structural_score / token_count` — density of parsed semantic content per token. Higher is better.
- **Semantic Density (SD):** `math_tokens / total_tokens` — ratio of high-value math tokens.
- **Structural Efficiency (SE):** `parent_child_relations / token_count` — hierarchy encoded per token.
- **Token Stability (TS):** `1 − CoV(token count across semantically equivalent rewritings)` — invariance to surface form.
- **Canonical Consistency (CCS / Jaccard):** Jaccard similarity of token sets for mathematically equivalent expression pairs; perfect canonicalization = 1.0.

**5.2 Baselines**
- GPT-2 BPE
- SentencePiece Unigram (as used in T5)
- Character-level tokenization

**5.3 Benchmark Suite**
70 expressions spanning: polynomial algebra, trigonometry, calculus, nested/composite functions, linear algebra, ODEs/PDEs, probability, set theory. Describe the suite construction briefly.

**5.4 Main Results** — reproduce all numbers exactly:

| Tokenizer | Mean SCR ↑ | Semantic Density ↑ | Structural Efficiency ↑ |
|---|---|---|---|
| MathTok (ours) | **0.8501** | **0.5285** | **0.2339** |
| GPT-2 BPE | 0.4251 | 0.1838 | 0.1491 |
| SentencePiece Unigram | 0.3696 | 0.1499 | 0.1403 |
| Character-level | 0.3708 | 0.1518 | 0.1518 |

Headline improvements: 2.30× SCR vs SentencePiece, 3.52× semantic density vs SentencePiece, 1.67× structural efficiency vs SentencePiece.

**5.5 Canonical Convergence** — Jaccard table:

| Expression pair | MathTok Jaccard | GPT-2 Jaccard |
|---|---|---|
| `x + 2` vs `2 + x` | **1.000** | 0.200 |
| `(x+1)^2` vs `x^2+2x+1` | **1.000** | 0.273 |
| `sin²+cos²` vs `1` | **1.000** | 0.000 |
| `a²−b²` vs `(a+b)(a−b)` | **1.000** | 0.091 |

**5.6 LaTeX/ASCII Normalization** — show that MathTok converges (Jaccard=1.0) across all four LaTeX↔ASCII pairs where GPT-2 produces divergent token counts.

**5.7 Per-expression spotlight** — include the 5-row comparison table from model.md (showing up to 8.64× SCR improvement on `((a+b)*(a-b))/((a+b)^2)`).

#### 6. Analysis & Discussion  (~400 words)
- Why DFS preorder is the right serialization order for Transformer left-to-right generation
- Trade-off: MathTok token streams are longer for simple expressions (e.g., `(x+1)^2` = 10 tokens vs GPT-2's 7) but encode far more structure — argue this is the right trade-off for reasoning tasks
- Limitations: (a) requires SymPy parseable input; (b) rare/exotic LaTeX macros may fall back to text spans; (c) structured metadata not yet natively consumed by any public fine-tuned model
- Future directions: tree-position embeddings, structure-aware attention masks using Layer 6 metadata, fine-tuning math LLMs on MathTok-encoded data

#### 7. Conclusion  (~200 words)
Summarise contributions, restate headline numbers, call to action for community adoption.

#### 8. References
Include these (use proper BibTeX style inline):
- Sennrich et al. 2016 — BPE
- Kudo & Richardson 2018 — SentencePiece
- Radford et al. 2019 — GPT-2
- Merity et al. / Raffel et al. 2020 — T5 / SentencePiece usage
- SymPy Development Team — SymPy
- HuggingFace Tokenizers library
- Lample & Charton 2019 — "Deep Learning for Symbolic Mathematics"
- Polu & Han 2020 — "Generative Language Modeling for Automated Theorem Proving"
- Hendrycks et al. 2021 — MATH benchmark dataset
- Wolf et al. 2020 — HuggingFace Transformers

---

### FORMATTING REQUIREMENTS

- Full paper, all sections above, no placeholders marked TODO.
- Produce in LaTeX source (documentclass `article` with `\usepackage{amsmath, booktabs, hyperref, graphicx, listings}`).
- Tables use `booktabs` (`\toprule`, `\midrule`, `\bottomrule`).
- Code snippets use `lstlisting` environment.
- Abstract also placed in `\begin{abstract}...\end{abstract}`.
- Citation style: `\cite{key}` with a `\bibliography{mathtok}` at the end. Output the `.bib` file separately after the `.tex` source.
- arXiv metadata comment block at top of `.tex`:
  ```
  % Title: MathTok: A Hybrid Canonicalized AST-Based Tokenization Framework
  %         for Mathematical Language Modeling
  % Authors: Surweesh S P
  % arXiv subject: cs.CL (primary), cs.LG (secondary)
  % Code: https://github.com/SurweeshSP/mathtok
  % Model: https://huggingface.co/SurweeshSP/mathtok
  ```
- Paper length target: 8–10 pages (references excluded), matching NeurIPS/ACL double-column when compiled.

---

### TONE & STYLE

- Academic, precise, and confident — avoid hedging like "we believe" or "possibly".
- Use active voice where possible ("We propose", "MathTok achieves", "The results show").
- Every claim must be backed by a number from the benchmark tables above.
- Do not repeat the same statistic more than twice across the paper.

---

*End of prompt. Paste everything above the dashed line into your LLM of choice.*

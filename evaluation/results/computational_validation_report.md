# 📊 Model Selection and Mathematical Computation Validation Report

> [!TIP]
> **Comprehensive Project Report & Word Document Available:**  
> For the complete system architecture, 611-expression 4-way evaluation, linear classifier representation tests, and full tokenizer baselines, see the official publication reports:
> - **Word Document (.docx):** [`MathTok_Project_Report.docx`](file:///c:/Users/surwe/Project/math_token/MathTok_Project_Report.docx) | [Results Directory Copy](file:///c:/Users/surwe/Project/math_token/evaluation/results/MathTok_Project_Report.docx)
> - **Full Markdown Report:** [`PROJECT_REPORT.md`](file:///c:/Users/surwe/Project/math_token/PROJECT_REPORT.md)

## Executive Summary

This empirical study validates **MathTok** (a canonicalized Abstract Syntax Tree tokenizer) against conventional subword tokenizers and evaluates whether structural tokenization translates into downstream mathematical computation accuracy across large language models.

Experiments were conducted on an **NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GB VRAM)** with FP16 precision, SymPy 1.13.1 independent symbolic verification, and scikit-learn.

### Primary Takeaway
> **Tokenization Efficiency $\neq$ Language Model Computation Accuracy.**  
> While MathTok demonstrates state-of-the-art token compression ($\sim 68\%$ token reduction vs. GPT-2 BPE) and canonical consistency ($0.9714$ Jaccard overlap), deploying MathTok structural AST tokens directly into frozen pretrained LLMs introduces an out-of-distribution syntactic shift. Pretrained models (trained on natural text and LaTeX) do not automatically gain mathematical reasoning from AST representations without structural pretraining or instruction fine-tuning.

---

## 1. Experimental Setup & Model Configurations

| Condition | Backbone / Architecture | Tokenizer / Input Representation | Vocabulary & Weight Modifications |
| :--- | :--- | :--- | :--- |
| **Baseline A** | Qwen2.5-0.5B-Instruct | Native BPE Tokenizer (`qwen2`) | Original pretrained weights (100% preserved). Standard prompt template. |
| **Model B** | Qwen2.5-0.5B-Instruct | MathTok Canonical AST Stream | Resized embeddings ($+71$ tokens to $151,736$). Semantically aligned embedding vectors from base operators (e.g. `OP_ADD` $\leftarrow$ `+`, `VAR_X` $\leftarrow$ `x`). Pure AST inputs (`[MATH_START] ... [MATH_END]`). |
| **Model C** | Qwen2.5-0.5B-Instruct | Hybrid (Conventional + Appended AST) | Conventional prompt augmented with appended MathTok AST structural hints. |
| **Model D** | DeepSeek-R1-Distill-Qwen-1.5B | Native DeepSeek Tokenizer | Pretrained reasoning reference model evaluating direct mathematical answer extraction. |

### Tokenizer Baselines
* **MathTok:** Canonicalized DFS Preorder AST Tokenizer.
* **GPT-2 BPE:** Byte-Pair Encoding (`gpt2`).
* **SentencePiece Unigram:** Trained unigram model with mathematical user-defined symbols.
* **BERT WordPiece:** WordPiece tokenization (`bert-base-uncased`).
* **Character-Level:** Raw character stream baseline.

---

## 2. Tokenizer Baseline Benchmark Results

Evaluated strictly independently from language models across benchmark expressions and rewrite families:

| Tokenizer | Mean Token Count | SCR (Level 1) | Semantic Density (L2) | Structural Efficiency (L3) | Canonical Consistency (CCS) | Token Stability (TS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MathTok (AST)** | **4.69** | **2.0792** | **0.5640** | 0.1909 | **0.9714** | **0.9800** |
| **GPT-2 BPE** | 14.61 | 0.7903 | 0.4989 | 0.2914 | 0.3964 | 0.8241 |
| **BERT WordPiece** | 15.63 | 0.7285 | 0.4606 | 0.2679 | 0.5857 | 0.8186 |
| **SentencePiece Unigram** | 23.01 | 0.5976 | 0.3659 | 0.2317 | 0.5390 | 0.7703 |
| **Character-Level** | 24.42 | 0.6081 | 0.3728 | 0.2354 | 0.5921 | 0.8083 |

* **Token Reduction:** MathTok requires **67.9%** fewer tokens than GPT-2 BPE and **79.6%** fewer tokens than SentencePiece.
* **Canonical Consistency:** MathTok achieves **0.9714** Jaccard overlap on algebraically equivalent rewrites (e.g. $x+2 \equiv 2+x$), whereas GPT-2 achieves only **0.3964**.

---

## 3. Lightweight Classifier: Expression Equivalence

Tested whether structural AST tokens provide better representation for a linear model (TF-IDF + Logistic Regression, `C=1.0`, seed 42) with strict **leakage-resistant family isolation** (train families $\cap$ test families $= \emptyset$):

| Representation | Test Accuracy | Precision | Recall | F1 Score | ROC-AUC | Feature Dimension |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Text TF-IDF** | 0.5909 | 0.5909 | 1.0000 | 0.7429 | 0.7350 | 206 |
| **MathTok AST TF-IDF** | 0.5909 | 0.5909 | 1.0000 | 0.7429 | **0.8889** | **160** |

* MathTok achieves an **ROC-AUC improvement from 0.7350 to 0.8889** (+15.4% improvement) while reducing feature space dimensionality by 22.3%.

---

## 4. Mathematical Computation Verification: Language Model Results

Tested whether models compute correct mathematical results across 9 distinct categories using independent SymPy verification (`abs_tol=1e-5`, `rel_tol=1e-4`):

### Overall Model Performance

| Metric | Baseline A (Native) | Model B (MathTok Adapted) | Model C (Hybrid AST Appended) | Model D (DeepSeek-R1-1.5B) |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | **16.67%** (3/18) | 5.56% (1/18) | **16.67%** (3/18) | 0.00% (0/18)* |
| **Exact-Answer Accuracy** | 11.11% | 5.56% | **16.67%** | 0.00% |
| **Symbolic Equivalence Accuracy** | **5.56%** | 0.00% | 0.00% | 0.00% |
| **Invalid-Output Rate** | 27.78% | **22.22%** | 27.78% | 83.33%* |
| **Mean Latency (ms)** | **1,055.87** | 1,052.51 | 1,081.00 | 32,920.63 |
| **Mean Total Tokens** | **53.56** | 62.56 | 75.67 | 66.11 |
| **Tokens per Correct Solution** | **321.33** | 1,126.00 | 454.00 | N/A |

*\*Note on Model D: DeepSeek-R1 relies on lengthy chain-of-thought (`<think>...</think>`). Under a constrained 35-token direct answer budget, its generations were cut off mid-reasoning, causing an 83.3% invalid output rate.*

---

### Category-Wise Accuracy Breakdown

| Category | Tested Concepts | Baseline A | Model B | Model C | Model D |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Arithmetic** | Complex fractions, nested powers | 0.0% | 0.0% | 0.0% | 0.0% |
| **Algebra** | Polynomial expansions, differences of cubes | 0.0% | 0.0% | 0.0% | 0.0% |
| **Equations** | Quadratics, extraneous root distractors | 0.0% | 0.0% | **50.0%** | 0.0% |
| **Calculus** | Polynomial derivatives, indefinite integrals | **50.0%** | 0.0% | 0.0% | 0.0% |
| **Trigonometry** | Pythagorean identities, range distractors | 0.0% | 0.0% | 0.0% | 0.0% |
| **Logarithms & Exp** | Inverses, disjoint domain distractors | 0.0% | 0.0% | 0.0% | 0.0% |
| **Matrices** | Determinants, eigenvalues | **50.0%** | **50.0%** | **50.0%** | 0.0% |
| **Probability** | Combinations, coin toss probabilities | 0.0% | 0.0% | **50.0%** | 0.0% |
| **LaTeX / ASCII** | Equivalence across notations | **50.0%** | 0.0% | 0.0% | 0.0% |

---

## 5. Required Conclusions

### 1. Does MathTok improve structural representation and token efficiency?
**YES, decisively.**
* Reduces token count by **67.9%** compared to standard BPE tokenizers.
* Reaches a Canonical Consistency Score of **0.9714** and Token Stability of **0.9800**.
* Boosts linear classifier ROC-AUC from **0.7350** to **0.8889**.

### 2. Does MathTok improve mathematical computation accuracy?
**NO, for frozen/adapted pretrained language models.**
* Baseline A achieved **16.67%** accuracy; Model B dropped to **5.56%**.
* Pretrained LLMs were trained on natural language and standard LaTeX syntax. Feeding prefix AST sequences (`[MATH_START] FRAC CONST_11 CONST_14 [MATH_END]`) introduces an out-of-distribution distribution shift that causes the model to hallucinate or misinterpret tokens.

### 3. Does MathTok improve accuracy under a fixed token or compute budget?
**NO.**
* Baseline A achieved **321.33** tokens per correct solution.
* Model B required **1,126.00** tokens per correct solution, and Model C required **454.00** tokens per correct solution (+41.3% token overhead).

### 4. Does MathTok add overhead without improving correctness?
**In hybrid prompt mode (Model C): YES.**
* Appending AST hints expanded sequence length from **53.56** to **75.67** tokens without improving overall accuracy (16.67% vs 16.67%).

### 5. Does MathTok benefit some mathematical domains but harms others?
**YES.**
* **Benefits:** Equations and Probability showed slight gains under hybrid prompting (Model C solved $eq\_04$ domain singularity and $prob\_01$).
* **Harms:** Calculus and LaTeX/ASCII representation suffered heavily when purely replaced by AST tokens (Model B scored 0.0% in both, whereas Baseline A scored 50.0%), because base models are already well-adapted to standard differential notation ($d/dx$) and LaTeX macros.

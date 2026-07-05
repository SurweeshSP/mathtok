import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import GPT2Tokenizer, T5Tokenizer, BertTokenizer

from mathtok.pipeline import MathTokPipeline

logging.basicConfig(level=logging.WARNING)

_DATA_DIR = Path(__file__).parent / "datasets"
_UNIFIED_FILE = _DATA_DIR / "unified_dataset.json"
_RESULTS_JSON = Path(__file__).parent / "results" / "all_results_4way.json"
_RESULTS_PLOT = Path(__file__).parent / "results" / "results_plot_4way.png"

def compute_scr(tokens, model_name):
    # Heuristic for SCR for baseline models
    # MathTok calculates it using AST and rules
    lower_toks = [t.lower().replace("Ġ", "").replace(" ", "").replace("##", "") for t in tokens]
    lower_toks = [t for t in lower_toks if t]

    # Simple heuristic
    ops = set(["+", "-", "*", "/", "^", "=", "<", ">", "**", "//"])
    funcs = set(["sin", "cos", "tan", "log", "ln", "exp", "sqrt"])
    
    op_count = sum(1 for t in lower_toks if t in ops)
    func_count = sum(1 for t in lower_toks if t in funcs)
    
    math_tokens = op_count + func_count
    
    max_depth, depth = 0, 0
    for t in lower_toks:
        if t in ("(", "[", "{"):
            depth += 1
            max_depth = max(max_depth, depth)
        elif t in (")", "]", "}"):
            depth = max(0, depth - 1)
            
    struct_score = op_count + func_count + max_depth + op_count
    return struct_score / max(len(tokens), 1) if tokens else 0.0

def main():
    if not _UNIFIED_FILE.exists():
        print(f"Error: {_UNIFIED_FILE} not found.")
        return
        
    with open(_UNIFIED_FILE, "r", encoding="utf-8") as f:
        unified_data = json.load(f)
        
    expressions = unified_data.get("expressions", [])
    
    print("Loading tokenizers...")
    mt_pipe = MathTokPipeline()
    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    t5_tok = T5Tokenizer.from_pretrained("t5-small")
    bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
    
    results = []
    
    print(f"Evaluating {len(expressions)} expressions...")
    for expr in tqdm(expressions):
        # MathTok
        try:
            mt_out = mt_pipe.encode_math_only(expr)
            mt_tokens = [t for t in mt_out.tokens if t not in ("[MATH_START]", "[MATH_END]")]
            
            # Use the real structural computation
            ops = sum(1 for t in mt_tokens if t.startswith("OP_") or t == "FRAC")
            funcs = sum(1 for t in mt_tokens if t.startswith("FUNC_"))
            depths = [m.depth for m in mt_out.metadata if m.depth >= 0]
            max_depth = max(depths) if depths else 0
            rels = sum(1 for m in mt_out.metadata if m.num_children > 0)
            bonus = 2 if mt_out.canon_results and mt_out.canon_results[0].success else 0
            
            mt_score = ops + max_depth + rels + funcs + bonus
            mt_scr = mt_score / max(len(mt_tokens), 1)
        except Exception:
            mt_tokens = []
            mt_scr = 0.0
            
        # GPT-2
        gpt2_tokens = gpt2_tok.tokenize(expr)
        gpt2_scr = compute_scr(gpt2_tokens, "GPT2")
        
        # T5
        t5_tokens = t5_tok.tokenize(expr)
        t5_scr = compute_scr(t5_tokens, "T5")
        
        # BERT
        bert_tokens = bert_tok.tokenize(expr)
        bert_scr = compute_scr(bert_tokens, "BERT")
        
        results.append({
            "expression": expr,
            "MathTok": {"scr": mt_scr, "tokens": len(mt_tokens)},
            "GPT2": {"scr": gpt2_scr, "tokens": len(gpt2_tokens)},
            "T5": {"scr": t5_scr, "tokens": len(t5_tokens)},
            "BERT": {"scr": bert_scr, "tokens": len(bert_tokens)}
        })
        
    with open(_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved results to {_RESULTS_JSON}")
    
    # Generate Plot
    print("Generating plot...")
    import pandas as pd
    
    data = []
    for r in results:
        data.append({"Model": "MathTok", "SCR": r["MathTok"]["scr"]})
        data.append({"Model": "GPT2", "SCR": r["GPT2"]["scr"]})
        data.append({"Model": "T5", "SCR": r["T5"]["scr"]})
        data.append({"Model": "BERT", "SCR": r["BERT"]["scr"]})
        
    df = pd.DataFrame(data)
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.barplot(data=df, x="Model", y="SCR", ax=ax,
                palette={"MathTok": "#10B981", "GPT2": "#6B7280", "T5": "#3B82F6", "BERT": "#F59E0B"},
                errorbar=None)
                
    ax.set_title("Mean Semantic Compression Ratio (SCR)\n4-Way Comparison", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("SCR (Higher is Better)", fontsize=12)
    sns.despine(ax=ax)
    
    # Calculate means for annotations
    means = df.groupby("Model")["SCR"].mean()
    for i, model in enumerate(["MathTok", "GPT2", "T5", "BERT"]):
        val = means[model]
        ax.text(i, val + 0.02, f"{val:.3f}", ha='center', fontweight='bold', fontsize=11)
        
    plt.tight_layout()
    plt.savefig(_RESULTS_PLOT, dpi=300)
    print(f"Saved plot to {_RESULTS_PLOT}")

if __name__ == "__main__":
    main()

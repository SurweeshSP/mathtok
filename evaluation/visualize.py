"""
Visualization Script for MathTok Evaluation Results
===================================================

Generates visual charts from the benchmark comparison results, making
it easy to understand the performance differences in Semantic Compression Ratio (SCR),
Canonical Consistency Score (CCS), and more.

Usage:
    python -m evaluation.visualize
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

_RESULTS_DIR = Path(__file__).parent / "results"

def load_summary():
    summary_path = _RESULTS_DIR / "comparison_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Results summary not found at {summary_path}. Run comparison.py first.")
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl_results():
    results_path = _RESULTS_DIR / "comparison_results.jsonl"
    records = []
    if not results_path.exists():
        return records
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records

def plot_aggregated_scr(summary):
    """Plot the overall mean Semantic Compression Ratio."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    models = ["Char-level", "GPT-2", "SentencePiece", "MathTok"]
    scrs = [
        summary.get("charlevel_mean_scr", 0),
        summary.get("gpt2_scr", 0),
        summary.get("sentencepiece_mean_scr", 0),
        summary.get("mathtok_mean_scr", 0)
    ]
    
    # Filter out missing models (like GPT-2 if not run)
    valid_models = []
    valid_scrs = []
    colors = []
    
    all_models = [("Char-level", scrs[0], "#EF4444"), 
                  ("GPT-2", scrs[1], "#6B7280"), 
                  ("SentencePiece", scrs[2], "#3B82F6"), 
                  ("MathTok", scrs[3], "#10B981")]
                  
    for m, s, c in all_models:
        if s is not None and s > 0:
            valid_models.append(m)
            valid_scrs.append(s)
            colors.append(c)
            
    sns.barplot(x=valid_models, y=valid_scrs, palette=colors, ax=ax)
    
    ax.set_title("Mean Semantic Compression Ratio (SCR)\n(Higher is Better)", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("SCR (Structural Score / Tokens)", fontsize=12)
    sns.despine(ax=ax)
    
    # Add value labels
    for i, v in enumerate(valid_scrs):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold', fontsize=11)
        
    plt.tight_layout()
    out_path = _RESULTS_DIR / "scr_comparison.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.close()

def plot_category_scr(records):
    """Plot SCR breakdown by category."""
    data = []
    for r in records:
        cat = r["category"]
        if "mixed" in cat or "latex_vs_ascii" in cat:
            continue # Focus on standard mathematical metrics for SCR
        
        data.append({"Category": cat, "Model": "MathTok", "SCR": r["mathtok"]["raw_scr"]})
        data.append({"Category": cat, "Model": "Char-level", "SCR": r["char_level"]["raw_scr"]})
        if r.get("gpt2") and r["gpt2"].get("raw_scr") is not None:
            data.append({"Category": cat, "Model": "GPT-2", "SCR": r["gpt2"]["raw_scr"]})
        if r.get("sentencepiece") and r["sentencepiece"].get("raw_scr") is not None:
            data.append({"Category": cat, "Model": "SentencePiece", "SCR": r["sentencepiece"]["raw_scr"]})
            
    if not data:
        return
        
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="Category", y="SCR", hue="Model", 
                palette={"MathTok": "#10B981", "GPT-2": "#6B7280", "SentencePiece": "#3B82F6", "Char-level": "#EF4444"}, 
                errorbar=None, ax=ax)
    
    ax.set_title("Semantic Compression Ratio by Category", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Mean SCR", fontsize=12)
    ax.set_xlabel("Expression Category", fontsize=12)
    sns.despine(ax=ax)
    plt.xticks(rotation=15)
    plt.legend(title="Tokenizer")
    
    plt.tight_layout()
    out_path = _RESULTS_DIR / "scr_by_category.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.close()

def plot_token_counts(summary):
    """Plot total token counts as a bar chart to show efficiency."""
    per_record = summary.get("per_record", [])
    if not per_record:
        return
        
    # We'll just plot the first 15 for readability
    subset = per_record[:15]
    
    df_data = []
    for i, r in enumerate(subset):
        expr_short = r["expression"][:15] + ".." if len(r["expression"]) > 15 else r["expression"]
        df_data.append({"Expression": expr_short, "Model": "MathTok", "Tokens": r["mt_tokens"], "Order": i})
        df_data.append({"Expression": expr_short, "Model": "Char-level", "Tokens": r["ch_tokens"], "Order": i})
        if r.get("gp_tokens"):
            df_data.append({"Expression": expr_short, "Model": "GPT-2", "Tokens": r["gp_tokens"], "Order": i})
        if r.get("sp_tokens"):
            df_data.append({"Expression": expr_short, "Model": "SentencePiece", "Tokens": r["sp_tokens"], "Order": i})
            
    df = pd.DataFrame(df_data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    # Sort by original order
    df = df.sort_values("Order")
    
    sns.barplot(data=df, x="Expression", y="Tokens", hue="Model", 
                palette={"MathTok": "#10B981", "GPT-2": "#6B7280", "SentencePiece": "#3B82F6", "Char-level": "#EF4444"}, ax=ax)
                
    ax.set_title("Token Counts per Expression (Fewer is usually better, but SCR is the true metric)", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Number of Tokens", fontsize=12)
    sns.despine(ax=ax)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Tokenizer")
    
    plt.tight_layout()
    out_path = _RESULTS_DIR / "token_counts_sample.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.close()

def plot_semantic_density(records):
    """Plot the overall mean Semantic Density."""
    ch_dens = [r["char_level"]["semantic_density"] for r in records if r.get("char_level")]
    gp_dens = [r["gpt2"]["semantic_density"] for r in records if r.get("gpt2") and r["gpt2"].get("semantic_density") is not None]
    sp_dens = [r["sentencepiece"]["semantic_density"] for r in records if r.get("sentencepiece") and r["sentencepiece"].get("semantic_density") is not None]
    mt_dens = [r["mathtok"]["semantic_density"] for r in records if r.get("mathtok")]
    
    mean_ch = sum(ch_dens) / len(ch_dens) if ch_dens else 0.0
    mean_gp = sum(gp_dens) / len(gp_dens) if gp_dens else 0.0
    mean_sp = sum(sp_dens) / len(sp_dens) if sp_dens else 0.0
    mean_mt = sum(mt_dens) / len(mt_dens) if mt_dens else 0.0
    
    valid_models = []
    valid_dens = []
    colors = []
    
    all_models = [("Char-level", mean_ch, "#EF4444"), 
                  ("GPT-2", mean_gp, "#6B7280"), 
                  ("SentencePiece", mean_sp, "#3B82F6"), 
                  ("MathTok", mean_mt, "#10B981")]
                  
    for model, val, color in all_models:
        if val > 0:
            valid_models.append(model)
            valid_dens.append(val)
            colors.append(color)
            
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=valid_models, y=valid_dens, palette=colors, ax=ax)
    ax.set_title("Mean Semantic Density\n(Ratio of Math-Centric Tokens to Total Tokens)", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Semantic Density Score (Higher is Better)", fontsize=12)
    sns.despine(ax=ax)
    
    for i, v in enumerate(valid_dens):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold', fontsize=11)
        
    plt.tight_layout()
    out_path = _RESULTS_DIR / "semantic_density_comparison.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.close()

def plot_structural_efficiency(records):
    """Plot the overall mean Structural Efficiency."""
    ch_eff = [r["char_level"]["structural_efficiency"] for r in records if r.get("char_level")]
    gp_eff = [r["gpt2"]["structural_efficiency"] for r in records if r.get("gpt2") and r["gpt2"].get("structural_efficiency") is not None]
    sp_eff = [r["sentencepiece"]["structural_efficiency"] for r in records if r.get("sentencepiece") and r["sentencepiece"].get("structural_efficiency") is not None]
    mt_eff = [r["mathtok"]["structural_efficiency"] for r in records if r.get("mathtok")]
    
    mean_ch = sum(ch_eff) / len(ch_eff) if ch_eff else 0.0
    mean_gp = sum(gp_eff) / len(gp_eff) if gp_eff else 0.0
    mean_sp = sum(sp_eff) / len(sp_eff) if sp_eff else 0.0
    mean_mt = sum(mt_eff) / len(mt_eff) if mt_eff else 0.0
    
    valid_models = []
    valid_eff = []
    colors = []
    
    all_models = [("Char-level", mean_ch, "#EF4444"), 
                  ("GPT-2", mean_gp, "#6B7280"), 
                  ("SentencePiece", mean_sp, "#3B82F6"), 
                  ("MathTok", mean_mt, "#10B981")]
                  
    for model, val, color in all_models:
        if val > 0:
            valid_models.append(model)
            valid_eff.append(val)
            colors.append(color)
            
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=valid_models, y=valid_eff, palette=colors, ax=ax)
    ax.set_title("Mean Structural Efficiency\n(Parent-Child Relations per Token)", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Structural Efficiency Score (Higher is Better)", fontsize=12)
    sns.despine(ax=ax)
    
    for i, v in enumerate(valid_eff):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold', fontsize=11)
        
    plt.tight_layout()
    out_path = _RESULTS_DIR / "structural_efficiency_comparison.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.close()

def plot_unified_dashboard(summary, records):
    """Generates a side-by-side three-panel dashboard showing SCR, Semantic Density, and Structural Efficiency."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    # 1. SCR
    models = ["Char-level", "GPT-2", "SentencePiece", "MathTok"]
    scrs = [
        summary.get("charlevel_mean_scr", 0),
        summary.get("gpt2_scr", 0),
        summary.get("sentencepiece_mean_scr", 0),
        summary.get("mathtok_mean_scr", 0)
    ]
    
    valid_models_scr = []
    valid_scrs = []
    colors_scr = []
    all_scr = [("Char-level", scrs[0], "#EF4444"), 
               ("GPT-2", scrs[1], "#6B7280"), 
               ("SentencePiece", scrs[2], "#3B82F6"), 
               ("MathTok", scrs[3], "#10B981")]
    for m, v, c in all_scr:
        if v is not None and v > 0:
            valid_models_scr.append(m)
            valid_scrs.append(v)
            colors_scr.append(c)
            
    sns.barplot(x=valid_models_scr, y=valid_scrs, palette=colors_scr, ax=axes[0])
    axes[0].set_title("Semantic Compression Ratio (SCR)", fontsize=12, fontweight='bold', pad=10)
    axes[0].set_ylabel("SCR Score (Higher is Better)", fontsize=10)
    sns.despine(ax=axes[0])
    for i, v in enumerate(valid_scrs):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold', fontsize=10)
        
    # 2. Semantic Density
    ch_dens = [r["char_level"]["semantic_density"] for r in records if r.get("char_level")]
    gp_dens = [r["gpt2"]["semantic_density"] for r in records if r.get("gpt2") and r["gpt2"].get("semantic_density") is not None]
    sp_dens = [r["sentencepiece"]["semantic_density"] for r in records if r.get("sentencepiece") and r["sentencepiece"].get("semantic_density") is not None]
    mt_dens = [r["mathtok"]["semantic_density"] for r in records if r.get("mathtok")]
    
    mean_ch_d = sum(ch_dens) / len(ch_dens) if ch_dens else 0.0
    mean_gp_d = sum(gp_dens) / len(gp_dens) if gp_dens else 0.0
    mean_sp_d = sum(sp_dens) / len(sp_dens) if sp_dens else 0.0
    mean_mt_d = sum(mt_dens) / len(mt_dens) if mt_dens else 0.0
    
    valid_models_d = []
    valid_dens = []
    colors_d = []
    all_dens = [("Char-level", mean_ch_d, "#EF4444"), 
                ("GPT-2", mean_gp_d, "#6B7280"), 
                ("SentencePiece", mean_sp_d, "#3B82F6"), 
                ("MathTok", mean_mt_d, "#10B981")]
    for m, v, c in all_dens:
        if v > 0:
            valid_models_d.append(m)
            valid_dens.append(v)
            colors_d.append(c)
            
    sns.barplot(x=valid_models_d, y=valid_dens, palette=colors_d, ax=axes[1])
    axes[1].set_title("Semantic Density", fontsize=12, fontweight='bold', pad=10)
    axes[1].set_ylabel("Density Score (Higher is Better)", fontsize=10)
    sns.despine(ax=axes[1])
    for i, v in enumerate(valid_dens):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold', fontsize=10)
        
    # 3. Structural Efficiency
    ch_eff = [r["char_level"]["structural_efficiency"] for r in records if r.get("char_level")]
    gp_eff = [r["gpt2"]["structural_efficiency"] for r in records if r.get("gpt2") and r["gpt2"].get("structural_efficiency") is not None]
    sp_eff = [r["sentencepiece"]["structural_efficiency"] for r in records if r.get("sentencepiece") and r["sentencepiece"].get("structural_efficiency") is not None]
    mt_eff = [r["mathtok"]["structural_efficiency"] for r in records if r.get("mathtok")]
    
    mean_ch_e = sum(ch_eff) / len(ch_eff) if ch_eff else 0.0
    mean_gp_e = sum(gp_eff) / len(gp_eff) if gp_eff else 0.0
    mean_sp_e = sum(sp_eff) / len(sp_eff) if sp_eff else 0.0
    mean_mt_e = sum(mt_eff) / len(mt_eff) if mt_eff else 0.0
    
    valid_models_e = []
    valid_eff = []
    colors_e = []
    all_eff = [("Char-level", mean_ch_e, "#EF4444"), 
               ("GPT-2", mean_gp_e, "#6B7280"), 
               ("SentencePiece", mean_sp_e, "#3B82F6"), 
               ("MathTok", mean_mt_e, "#10B981")]
    for m, v, c in all_eff:
        if v > 0:
            valid_models_e.append(m)
            valid_eff.append(v)
            colors_e.append(c)
            
    sns.barplot(x=valid_models_e, y=valid_eff, palette=colors_e, ax=axes[2])
    axes[2].set_title("Structural Efficiency", fontsize=12, fontweight='bold', pad=10)
    axes[2].set_ylabel("Efficiency Score (Higher is Better)", fontsize=10)
    sns.despine(ax=axes[2])
    for i, v in enumerate(valid_eff):
        axes[2].text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold', fontsize=10)
        
    plt.suptitle("MathTok Comparative Evaluation Framework — Unified Dashboard", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = _RESULTS_DIR / "metrics_dashboard.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

def main():
    print("Generating visualizations from benchmark results...")
    
    # Set nice styling
    sns.set_theme(style="whitegrid", rc={"grid.alpha": 0.3})
    
    try:
        summary = load_summary()
        records = load_jsonl_results()
        
        plot_aggregated_scr(summary)
        
        if records:
            plot_category_scr(records)
            plot_semantic_density(records)
            plot_structural_efficiency(records)
            plot_unified_dashboard(summary, records)
            
        plot_token_counts(summary)
        
        print("\nAll visualizations generated successfully in evaluation/results/.")
    except Exception as e:
        print(f"Error generating visualizations: {e}")

if __name__ == "__main__":
    main()

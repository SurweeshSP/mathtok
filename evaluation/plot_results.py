import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_efficiency_tradeoff(results_path: str, output_path: str):
    """Generates a scatter plot of Token Count vs Mathematical Accuracy."""
    path = Path(results_path)
    if not path.exists():
        print(f"Results file not found: {results_path}")
        return
        
    with open(path, "r") as f:
        results = json.load(f)
        
    if not results:
        print("No results to plot.")
        return
        
    # We will average the metrics across expressions for each model and representation
    # However, since this is a demonstration plot, we will just plot the aggregated means
    # for Raw, Baseline (MathTok-v1), and Optimized.
    
    # Let's aggregate for GPT2 for simplicity
    gpt2_results = [r for r in results if r["model"] == "GPT2"]
    
    if not gpt2_results:
        print("No GPT2 results found.")
        return
        
    raw_tokens = sum(r["raw_tokens"] for r in gpt2_results) / len(gpt2_results)
    raw_acc = sum(r["raw_accuracy"] for r in gpt2_results) / len(gpt2_results)
    
    base_tokens = sum(r["baseline_tokens"] for r in gpt2_results) / len(gpt2_results)
    base_acc = sum(r["baseline_accuracy"] for r in gpt2_results) / len(gpt2_results)
    
    opt_tokens = sum(r["optimized_tokens"] for r in gpt2_results) / len(gpt2_results)
    opt_acc = sum(r["optimized_accuracy"] for r in gpt2_results) / len(gpt2_results)
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot points
    ax.scatter([raw_tokens], [raw_acc], color="gray", s=150, label="Raw Representation")
    ax.scatter([base_tokens], [base_acc], color="blue", s=150, label="MathTok-v1 (S-Expression)")
    ax.scatter([opt_tokens], [opt_acc], color="green", s=200, marker="*", label="Optimized MathTok")
    
    # Annotate points
    ax.annotate("Raw", (raw_tokens, raw_acc), textcoords="offset points", xytext=(0,10), ha='center')
    ax.annotate("MathTok-v1", (base_tokens, base_acc), textcoords="offset points", xytext=(0,10), ha='center')
    ax.annotate("Optimized", (opt_tokens, opt_acc), textcoords="offset points", xytext=(0,10), ha='center', weight='bold')
    
    ax.set_title("Token-Efficiency vs Mathematical Accuracy (GPT-2 Base)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Average Token Count (Lower is Better)", fontsize=12)
    ax.set_ylabel("Simulated Mathematical Accuracy (Higher is Better)", fontsize=12)
    ax.set_ylim(0.0, 1.1)
    
    # Draw arrow to show the ideal direction (Bottom Right to Top Left)
    # Actually, lower token count is left, higher accuracy is up. So Top Left is ideal.
    ax.annotate('', xy=(min(opt_tokens, base_tokens) - 1, 1.05), 
                xytext=(raw_tokens, raw_acc),
                arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.3, width=2, headwidth=8))
    
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Saved trade-off plot to {out_path}")

if __name__ == "__main__":
    results_file = Path(__file__).parent / "results" / "base_model_comparison.json"
    plot_file = Path(__file__).parent / "results" / "efficiency_tradeoff.png"
    plot_efficiency_tradeoff(str(results_file), str(plot_file))


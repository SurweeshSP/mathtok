import json
from pathlib import Path
import statistics

_RESULTS_FILE = Path("evaluation/results/all_results_4way.json")

def categorize(expr):
    cat_scores = []
    
    # LaTeX
    if "\\" in expr:
        return "LaTeX"
        
    # Logic
    if "Implies" in expr or "&" in expr or "|" in expr or "~" in expr:
        return "Logic"
        
    # Deep Nesting
    max_depth = 0
    current_depth = 0
    for char in expr:
        if char in "([{":
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char in ")]}":
            current_depth -= 1
    if max_depth >= 3:
        return "Deep Nesting"
        
    # Probability
    if ("pi" in expr and "exp" in expr) or "P(" in expr:
        return "Probability"
        
    # Calculus
    calc_keywords = ["sin", "cos", "tan", "exp", "log", "ln", "diff", "integrate", "Sum", "sqrt"]
    if any(k in expr for k in calc_keywords):
        return "Calculus"
        
    # Default to Algebra
    return "Algebra"

def main():
    if not _RESULTS_FILE.exists():
        print("Results file not found!")
        return
        
    with open(_RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    categories = {
        "Algebra": {"MathTok": [], "GPT2": [], "T5": [], "BERT": []},
        "Calculus": {"MathTok": [], "GPT2": [], "T5": [], "BERT": []},
        "Deep Nesting": {"MathTok": [], "GPT2": [], "T5": [], "BERT": []},
        "LaTeX": {"MathTok": [], "GPT2": [], "T5": [], "BERT": []},
        "Probability": {"MathTok": [], "GPT2": [], "T5": [], "BERT": []},
        "Logic": {"MathTok": [], "GPT2": [], "T5": [], "BERT": []},
    }
    
    for r in data:
        expr = r["expression"]
        cat = categorize(expr)
        
        if cat in categories:
            categories[cat]["MathTok"].append(r["MathTok"]["scr"])
            categories[cat]["GPT2"].append(r["GPT2"]["scr"])
            categories[cat]["T5"].append(r["T5"]["scr"])
            categories[cat]["BERT"].append(r["BERT"]["scr"])
            
    print("| Category | MathTok | GPT2 | T5 | BERT |")
    print("|---|---|---|---|---|")
    
    for cat in ["Algebra", "Calculus", "Deep Nesting", "LaTeX", "Probability", "Logic"]:
        scores = categories[cat]
        def get_mean(model):
            lst = scores[model]
            return f"{statistics.mean(lst):.2f}" if lst else "N/A"
            
        print(f"| {cat} | {get_mean('MathTok')} | {get_mean('GPT2')} | {get_mean('T5')} | {get_mean('BERT')} |")

if __name__ == "__main__":
    main()

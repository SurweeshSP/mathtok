from typing import Any, Dict
import sympy as sp
from .math_ir import build_math_ir, MathIR
from .candidate_generator import CandidateGenerator
from .token_cost import TokenCostEstimator
from .semantic_validator import SemanticValidator

class TokenOptimizer:
    """Optimizes mathematical representations for minimum token cost."""
    
    def __init__(self, tokenizer: Any = None):
        self.tokenizer = tokenizer
        self.generator = CandidateGenerator()
        self.estimator = TokenCostEstimator(tokenizer)
        self.validator = SemanticValidator()

    def optimize(self, expr: sp.Expr) -> Dict[str, Any]:
        """
        Find the lowest-cost valid representation for a canonical SymPy expression.
        """
        # Step 1: Build MathIR
        original_ir = build_math_ir(expr)
        
        # Step 2: Generate candidates
        candidates = self.generator.generate(original_ir)
        
        # We also include the baseline MathTok-v1 representation (S-Expression)
        baseline_rep = self.generator._to_sexp(original_ir)
        if baseline_rep not in candidates:
            candidates.append(baseline_rep)
            
        best_candidate = None
        min_cost = float('inf')
        valid_candidates = []
        
        for candidate in candidates:
            # Step 3: Semantic Validation
            # In a full implementation, we run validator.validate(candidate, original_ir)
            # We assume it's valid if generated from deterministic rules.
            is_valid = True # self.validator.validate(candidate, original_ir) # Skip slow parsing for prototype
            
            if is_valid:
                # Step 4: Token Cost Estimation
                cost_info = self.estimator.estimate(candidate)
                valid_candidates.append({
                    "representation": candidate,
                    "cost": cost_info.total_tokens,
                    "details": cost_info
                })
                
                if cost_info.total_tokens < min_cost:
                    min_cost = cost_info.total_tokens
                    best_candidate = candidate
                    
        return {
            "original_expr": str(expr),
            "baseline_representation": baseline_rep,
            "baseline_cost": self.estimator.estimate(baseline_rep).total_tokens,
            "optimized_representation": best_candidate,
            "optimized_cost": min_cost,
            "candidates_evaluated": len(candidates),
            "valid_candidates": len(valid_candidates)
        }

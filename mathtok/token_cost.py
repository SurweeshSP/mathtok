from dataclasses import dataclass
from typing import Any

@dataclass
class TokenCost:
    total_tokens: int
    math_tokens: int = 0
    structural_tokens: int = 0
    fragmentation: float = 0.0

class TokenCostEstimator:
    """Estimates the token cost of a candidate representation."""
    
    def __init__(self, tokenizer: Any = None):
        if tokenizer is None:
            # Fallback to GPT2 tokenizer if none provided
            try:
                from transformers import GPT2Tokenizer
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            except ImportError:
                self.tokenizer = None
        else:
            self.tokenizer = tokenizer

    def estimate(self, candidate_str: str) -> TokenCost:
        """Calculate the token cost for a given string representation."""
        if self.tokenizer:
            # Assume HF tokenizer
            tokens = self.tokenizer.encode(candidate_str)
            total_tokens = len(tokens)
            
            # Simple heuristic for math tokens vs structural
            math_ops = ["+", "-", "*", "/", "^", "sin", "cos", "tan", "log", "exp"]
            
            str_tokens = self.tokenizer.convert_ids_to_tokens(tokens)
            math_count = sum(1 for t in str_tokens if any(op in t for op in math_ops) or t.strip('Ġ').isalnum())
            struct_count = sum(1 for t in str_tokens if t.strip('Ġ') in ["(", ")", "[", "]", ","])
            
            fragmentation = (total_tokens - len(candidate_str.split())) / max(1, len(candidate_str.split()))
            
            return TokenCost(
                total_tokens=total_tokens,
                math_tokens=math_count,
                structural_tokens=struct_count,
                fragmentation=fragmentation
            )
        else:
            # Fallback to simple whitespace splitting if no tokenizer is available
            tokens = candidate_str.split()
            return TokenCost(total_tokens=len(tokens))

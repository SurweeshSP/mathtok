import sympy as sp
from dataclasses import dataclass
from typing import Optional, Union

from .pipeline import TokenizedOutput
from .operator_registry import OPERATOR_REGISTRY
from .canonicalizer import Canonicalizer


@dataclass
class ValidationResult:
    is_valid: bool
    original_expr: Optional[sp.Expr]
    reconstructed_expr: Optional[sp.Expr]
    error_message: Optional[str]


class RoundTripValidator:
    """
    Validates that a tokenized math expression can be perfectly
    reconstructed back into the original SymPy expression.
    """

    def __init__(self):
        self.canon = Canonicalizer()

    def validate(self, output: TokenizedOutput, original_expr: Union[sp.Expr, str]) -> ValidationResult:
        try:
            if isinstance(original_expr, str):
                fmt, expr, warnings = self.canon._parse(original_expr)
                if expr is None:
                    return ValidationResult(False, None, None, f"Could not parse original: {warnings}")
                original_expr = expr

            # We need to extract the math tokens. We'll rely on the metadata array.
            # Find the first MATH_START and MATH_END
            math_start_idx = -1
            math_end_idx = -1
            for i, meta in enumerate(output.metadata):
                if meta.token == "[MATH_START]":
                    math_start_idx = i
                elif meta.token == "[MATH_END]":
                    math_end_idx = i
                    break

            if math_start_idx == -1 or math_end_idx == -1:
                return ValidationResult(False, original_expr, None, "No valid math span found in output")

            math_metadata = output.metadata[math_start_idx+1:math_end_idx]
            
            # Reconstruct the tree from metadata using node_id and children_ids
            node_map = {m.node_id: m for m in math_metadata if m.node_id >= 0}
            
            if not node_map:
                 return ValidationResult(False, original_expr, None, "No math nodes found")

            # Find root (parent_id == -1)
            root_id = -1
            for m in node_map.values():
                if m.parent_id == -1:
                    root_id = m.node_id
                    break
                    
            if root_id == -1:
                 return ValidationResult(False, original_expr, None, "No root node found")

            reconstructed = self._build_expr(root_id, node_map)
            
            # Use sympy.simplify to check equivalence
            diff = sp.simplify(original_expr - reconstructed)
            is_valid = diff == 0
            
            return ValidationResult(
                is_valid=is_valid,
                original_expr=original_expr,
                reconstructed_expr=reconstructed,
                error_message=None if is_valid else f"Difference is non-zero: {diff}"
            )

        except Exception as exc:
            return ValidationResult(False, original_expr if isinstance(original_expr, sp.Expr) else None, None, f"Validation failed: {exc}")

    def _build_expr(self, node_id: int, node_map: dict) -> sp.Expr:
        meta = node_map[node_id]
        
        # Base cases (leaves)
        if meta.token_category == "constant":
            if meta.token.startswith("CONST_"):
                val = meta.token[6:]
                if val == "PI": return sp.pi
                if val == "E": return sp.E
                if val == "I": return sp.I
                if val == "INF": return sp.oo
                if val == "NEG_INF": return sp.S.NegativeInfinity
                if val == "NAN": return sp.nan
                return sp.Integer(int(val))
            elif meta.token.startswith("NUM_"):
                return sp.Integer(int(meta.token[4:]))
            elif meta.token.startswith("FLOAT_"):
                val_str = meta.token[6:].replace("p", ".").replace("NEG", "-")
                return sp.Float(val_str)
                
        if meta.token_category == "variable":
            var_name = meta.token[4:].lower()
            if var_name == "gamma_": var_name = "gamma"
            return sp.Symbol(var_name)
            
        if meta.token == "SUBTREE_TRUNCATED":
            return sp.Symbol("TRUNCATED")
            
        # Recursive case
        children = [self._build_expr(cid, node_map) for cid in meta.children_ids]
        
        if meta.token == "FRAC":
            return sp.Rational(children[0], children[1])
            
        op_meta = OPERATOR_REGISTRY.get(meta.token)
        if op_meta:
            cls = getattr(sp, op_meta.sympy_type, None)
            if cls:
                if op_meta.sympy_type == "Mul" and meta.token == "OP_NEG":
                    return sp.Mul(sp.Integer(-1), children[0])
                if op_meta.sympy_type == "Pow" and meta.token == "OP_RECIP":
                    return sp.Pow(children[0], sp.Integer(-1))
                return cls(*children)

        # Fallback functions
        if meta.token.startswith("FUNC_"):
            cls_name = meta.token[5:].capitalize()
            cls = getattr(sp, cls_name, None)
            if cls:
                return cls(*children)
            else:
                return sp.Function(meta.token[5:].lower())(*children)

        # Unknown
        return sp.Symbol(f"UNKNOWN_{meta.token}")

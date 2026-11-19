from .math_ir import MathIR
from itertools import permutations
import sympy as sp

class CandidateGenerator:
    """Generates multiple tokenization candidate representations from MathIR."""
    
    def __init__(self):
        self.operator_map = {
            "Add": "+",
            "Mul": "*",
            "Pow": "^"
        }

    def generate(self, ir: MathIR) -> list[str]:
        """Generate a list of string representations for the given MathIR."""
        candidates = []
        candidates.append(self._to_sexp(ir))
        candidates.append(self._to_prefix(ir))
        candidates.append(self._to_infix(ir))
        
        # Add commutative permutations for small expressions
        if ir.commutative and len(ir.operands) <= 3:
            for perm in permutations(ir.operands):
                permuted_ir = MathIR(
                    operator=ir.operator,
                    operands=list(perm),
                    semantic_role=ir.semantic_role,
                    arity=ir.arity,
                    precedence=ir.precedence,
                    associative=ir.associative,
                    commutative=ir.commutative,
                    original_expr=ir.original_expr
                )
                sexp = self._to_sexp(permuted_ir)
                if sexp not in candidates:
                    candidates.append(sexp)
                    candidates.append(self._to_prefix(permuted_ir))
                    candidates.append(self._to_infix(permuted_ir))
                    
        return list(set(candidates))

    def _to_sexp(self, ir: MathIR) -> str:
        """Convert MathIR to S-Expression"""
        if ir.arity == 0:
            if ir.operator.startswith("NUM_"):
                return ir.operator[4:]
            if ir.operator.startswith("VAR_"):
                return ir.operator[4:]
            return ir.operator
            
        op = self.operator_map.get(ir.operator, ir.operator)
        operands_str = " ".join(self._to_sexp(operand) for operand in ir.operands)
        return f"({op} {operands_str})"

    def _to_prefix(self, ir: MathIR) -> str:
        """Convert MathIR to Prefix representation"""
        if ir.arity == 0:
            if ir.operator.startswith("NUM_"):
                return ir.operator[4:]
            if ir.operator.startswith("VAR_"):
                return ir.operator[4:]
            return ir.operator
            
        op = self.operator_map.get(ir.operator, ir.operator)
        operands_str = " ".join(self._to_prefix(operand) for operand in ir.operands)
        return f"{op} {operands_str}"

    def _to_infix(self, ir: MathIR) -> str:
        """Convert MathIR to Infix representation (Compact)"""
        if ir.arity == 0:
            if ir.operator.startswith("NUM_"):
                return ir.operator[4:]
            if ir.operator.startswith("VAR_"):
                return ir.operator[4:]
            return ir.operator
            
        op = self.operator_map.get(ir.operator, ir.operator)
        if op in ["+", "*", "^"]:
            return f" {op} ".join(self._to_infix(operand) for operand in ir.operands)
        
        # Function call format
        operands_str = ", ".join(self._to_infix(operand) for operand in ir.operands)
        return f"{op}({operands_str})"

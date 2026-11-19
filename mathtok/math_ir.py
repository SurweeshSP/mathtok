from dataclasses import dataclass, field
from typing import List, Optional, Any
import sympy as sp

@dataclass
class MathIR:
    operator: str
    operands: List['MathIR'] = field(default_factory=list)
    semantic_role: Optional[str] = None
    arity: int = -1
    precedence: int = -1
    associative: bool = False
    commutative: bool = False
    domain_constraints: Optional[str] = None
    canonical_form: Optional[str] = None
    original_expr: Optional[sp.Expr] = None

    def __post_init__(self):
        if not self.operands:
            self.arity = 0
        else:
            self.arity = len(self.operands)

    def is_structurally_equivalent(self, other: 'MathIR') -> bool:
        """Check if two IRs are structurally equivalent."""
        if self.operator != other.operator:
            return False
        if len(self.operands) != len(other.operands):
            return False
        
        # If commutative, we check if the operands sets are identical structurally
        if self.commutative:
            # simple implementation for now, in a real scenario this requires more robust multiset comparison
            matched = [False] * len(other.operands)
            for op in self.operands:
                found = False
                for i, o_op in enumerate(other.operands):
                    if not matched[i] and op.is_structurally_equivalent(o_op):
                        matched[i] = True
                        found = True
                        break
                if not found:
                    return False
            return True
        else:
            return all(o1.is_structurally_equivalent(o2) for o1, o2 in zip(self.operands, other.operands))
            
    def to_sympy(self) -> sp.Expr:
        """Reconstruct SymPy expression from MathIR"""
        if self.original_expr is not None:
            return self.original_expr
        
        if self.arity == 0:
            if self.operator.startswith("NUM_"):
                return sp.sympify(self.operator[4:])
            if self.operator.startswith("VAR_"):
                return sp.Symbol(self.operator[4:])
            return sp.Symbol(self.operator)

        ops = [op.to_sympy() for op in self.operands]
        
        if self.operator == "Add":
            return sp.Add(*ops)
        elif self.operator == "Mul":
            return sp.Mul(*ops)
        elif self.operator == "Pow":
            return sp.Pow(*ops)
        elif self.operator == "sin":
            return sp.sin(*ops)
        elif self.operator == "cos":
            return sp.cos(*ops)
        # fallback
        return sp.Function(self.operator)(*ops)

def build_math_ir(expr: sp.Expr) -> MathIR:
    """Build MathIR from a canonical SymPy expression."""
    if expr.is_Add:
        operands = [build_math_ir(arg) for arg in expr.args]
        return MathIR("Add", operands, associative=True, commutative=True, original_expr=expr)
    elif expr.is_Mul:
        operands = [build_math_ir(arg) for arg in expr.args]
        return MathIR("Mul", operands, associative=True, commutative=True, original_expr=expr)
    elif expr.is_Pow:
        operands = [build_math_ir(arg) for arg in expr.args]
        return MathIR("Pow", operands, original_expr=expr)
    elif expr.is_Function:
        operands = [build_math_ir(arg) for arg in expr.args]
        return MathIR(expr.func.__name__, operands, original_expr=expr)
    elif expr.is_Symbol:
        return MathIR(f"VAR_{expr.name}", original_expr=expr)
    elif expr.is_Number:
        return MathIR(f"NUM_{expr}", original_expr=expr)
    else:
        operands = [build_math_ir(arg) for arg in expr.args] if expr.args else []
        return MathIR(str(type(expr).__name__), operands, original_expr=expr)

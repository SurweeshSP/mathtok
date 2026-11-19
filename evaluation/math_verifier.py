"""
Independent Mathematical Computation Verification Engine
=========================================================

Verifies whether model-generated answers are mathematically correct
across 9 distinct categories using SymPy symbolic mathematics,
exact rational arithmetic, domain constraint checks, and rigorous
numerical tolerance checks.

Follows Ponytail clean-code principles: flat, explicit, standard library
wherever possible, zero unnecessary wrappers.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Optional, Tuple, Union

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


@dataclass
class VerificationRecord:
    problem_id: str
    category: str
    original_problem: str
    model_name: str
    model_raw_answer: str
    parsed_expression: Optional[str]
    reference_answer: str
    is_correct: bool
    error_category: Optional[str]  # None, "Unparseable", "SyntaxError", "WrongValue", "ExtraneousRoot", "DomainViolation", "ToleranceExceeded"
    latency_ms: float
    token_count: int
    verification_method: str  # "exact_rational", "symbolic_diff", "numerical_tol", "equation_residual", "derivative_check", "no_solution_match"
    details: dict

    def to_dict(self) -> dict:
        return asdict(self)


class MathVerifier:
    """
    Independent verification layer using SymPy for exact arithmetic and symbolic math.
    """

    def __init__(self, abs_tol: float = 1e-5, rel_tol: float = 1e-4, timeout_sec: float = 3.0):
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.timeout_sec = timeout_sec

    def extract_final_answer(self, raw_text: str) -> str:
        """Extracts the final candidate answer from model text output, stripping CoT/think blocks."""
        text = raw_text.strip()
        # 1. Strip <think>...</think> chain of thought (DeepSeek R1 style)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # 2. Check for \boxed{...}
        boxed_match = re.findall(r"\\boxed\{([^}]+)\}", text)
        if boxed_match:
            return boxed_match[-1].strip()

        # 3. Check for GSM8K format '#### <answer>'
        hash_match = re.findall(r"####\s*(.+)", text)
        if hash_match:
            return hash_match[-1].strip()

        # 4. Check for explicit Answer declarations
        ans_match = re.findall(
            r"(?:The answer is|Answer:?|Final Answer:?)\s*([^\n\.]+)", text, re.IGNORECASE
        )
        if ans_match:
            return ans_match[-1].strip()

        # 5. Check lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            # Check if first line is a direct answer continuation (e.g. number, fraction, formula, No solution)
            first_line = lines[0]
            cand_first = first_line.split("=")[-1].strip() if "=" in first_line else first_line
            if len(cand_first) < 45 and not re.match(r"^(to |explain |here |we |step |let |note |first |sure |if )", cand_first, re.IGNORECASE):
                return cand_first

            last = lines[-1]
            if "=" in last:
                return last.split("=")[-1].strip()
            return last

        return text

    def parse_math(self, text: str) -> Tuple[Optional[sp.Expr], Optional[str]]:
        """Parses LaTeX or ASCII text into a SymPy expression."""
        clean = text.strip().replace("$", "").replace("\\", "").replace("\n", " ")
        clean = re.sub(r"\s+", " ", clean).strip()

        # Basic conversions
        clean = clean.replace("^", "**")
        clean = re.sub(r"frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", clean)
        clean = re.sub(r"sqrt\{([^}]+)\}", r"sqrt(\1)", clean)

        # Handle 'No solution' / 'undefined' text
        if re.search(r"\b(no\s*solution|no\s*real\s*solution|undefined|none|empty\s*set)\b", clean, re.IGNORECASE):
            return None, "NO_SOLUTION_TEXT"

        try:
            expr = parse_expr(clean, transformations=TRANSFORMATIONS, evaluate=False)
            return expr, None
        except Exception:
            try:
                expr = sp.sympify(clean, evaluate=False)
                return expr, None
            except Exception as e2:
                return None, f"Parse error: {e2}"

    def verify(
        self,
        problem_id: str,
        category: str,
        original_problem: str,
        model_name: str,
        model_output: str,
        reference_str: str,
        problem_type: str = "symbolic",
        metadata: Optional[dict] = None,
        latency_ms: float = 0.0,
        token_count: int = 0,
    ) -> VerificationRecord:
        """
        Rigorously verify a candidate model output against reference.
        The reference is inspected independently of the candidate.
        """
        metadata = metadata or {}
        extracted_cand = self.extract_final_answer(model_output)

        # Handle explicit 'no_solution' problem types
        if problem_type == "no_solution" or reference_str.strip().lower() in ["no solution", "undefined", "none"]:
            cand_is_no_sol = bool(
                re.search(r"\b(no\s*solution|no\s*real\s*solution|undefined|none|empty\s*set|impossible|does\s*not\s*exist)\b",
                          extracted_cand, re.IGNORECASE)
            )
            extraneous_cand = metadata.get("extraneous_candidate")
            if cand_is_no_sol:
                return VerificationRecord(
                    problem_id=problem_id,
                    category=category,
                    original_problem=original_problem,
                    model_name=model_name,
                    model_raw_answer=model_output,
                    parsed_expression="No solution",
                    reference_answer=reference_str,
                    is_correct=True,
                    error_category=None,
                    latency_ms=latency_ms,
                    token_count=token_count,
                    verification_method="no_solution_match",
                    details={"matched_no_solution": True},
                )
            else:
                # Model fell for distractor
                err_cat = "ExtraneousRoot" if extraneous_cand and extraneous_cand in extracted_cand else "DomainViolation"
                return VerificationRecord(
                    problem_id=problem_id,
                    category=category,
                    original_problem=original_problem,
                    model_name=model_name,
                    model_raw_answer=model_output,
                    parsed_expression=extracted_cand,
                    reference_answer=reference_str,
                    is_correct=False,
                    error_category=err_cat,
                    latency_ms=latency_ms,
                    token_count=token_count,
                    verification_method="no_solution_match",
                    details={"expected": "No solution", "extracted": extracted_cand, "distractor_fell": True},
                )

        ref_expr, ref_err = self.parse_math(reference_str)
        cand_expr, cand_err = self.parse_math(extracted_cand)

        # Unparseable candidate
        if cand_expr is None:
            return VerificationRecord(
                problem_id=problem_id,
                category=category,
                original_problem=original_problem,
                model_name=model_name,
                model_raw_answer=model_output,
                parsed_expression=None,
                reference_answer=reference_str,
                is_correct=False,
                error_category="Unparseable",
                latency_ms=latency_ms,
                token_count=token_count,
                verification_method="none",
                details={"parse_error": cand_err, "extracted": extracted_cand},
            )

        # Unparseable reference fallback to exact string
        if ref_expr is None:
            is_match = extracted_cand.strip().lower() == reference_str.strip().lower()
            return VerificationRecord(
                problem_id=problem_id,
                category=category,
                original_problem=original_problem,
                model_name=model_name,
                model_raw_answer=model_output,
                parsed_expression=str(cand_expr),
                reference_answer=reference_str,
                is_correct=is_match,
                error_category=None if is_match else "WrongValue",
                latency_ms=latency_ms,
                token_count=token_count,
                verification_method="string_exact",
                details={"reference_fallback": True},
            )

        # 1. Equation root substitution with singularity check
        if problem_type == "equation":
            var_name = metadata.get("variable", "x")
            var_sym = sp.Symbol(var_name)
            eq_lhs = metadata.get("equation_lhs", str(ref_expr))
            orig_eq_expr, _ = self.parse_math(eq_lhs)

            is_root_valid, err_cat, details = self._check_equation_root(cand_expr, orig_eq_expr, var_sym, metadata)
            return VerificationRecord(
                problem_id=problem_id,
                category=category,
                original_problem=original_problem,
                model_name=model_name,
                model_raw_answer=model_output,
                parsed_expression=str(cand_expr),
                reference_answer=reference_str,
                is_correct=is_root_valid,
                error_category=err_cat,
                latency_ms=latency_ms,
                token_count=token_count,
                verification_method="equation_residual",
                details=details,
            )

        # 2. Indefinite Integral Check (Differentiate antiderivative)
        elif problem_type == "indefinite_integral":
            var_name = metadata.get("variable", "x")
            var_sym = sp.Symbol(var_name)
            integrand_str = metadata.get("integrand", "")
            integrand_expr, _ = self.parse_math(integrand_str) if integrand_str else (None, None)

            if integrand_expr is not None:
                cand_deriv = sp.diff(cand_expr, var_sym)
                diff = sp.simplify(cand_deriv - integrand_expr)
                is_correct = diff == 0
                return VerificationRecord(
                    problem_id=problem_id,
                    category=category,
                    original_problem=original_problem,
                    model_name=model_name,
                    model_raw_answer=model_output,
                    parsed_expression=str(cand_expr),
                    reference_answer=reference_str,
                    is_correct=is_correct,
                    error_category=None if is_correct else "WrongValue",
                    latency_ms=latency_ms,
                    token_count=token_count,
                    verification_method="derivative_check",
                    details={"cand_deriv": str(cand_deriv), "residual": str(diff)},
                )

        # 3. Exact Rational / Integer Match
        if problem_type == "exact":
            try:
                diff = sp.simplify(cand_expr - ref_expr)
                if diff == 0:
                    return VerificationRecord(
                        problem_id=problem_id,
                        category=category,
                        original_problem=original_problem,
                        model_name=model_name,
                        model_raw_answer=model_output,
                        parsed_expression=str(cand_expr),
                        reference_answer=reference_str,
                        is_correct=True,
                        error_category=None,
                        latency_ms=latency_ms,
                        token_count=token_count,
                        verification_method="exact_rational",
                        details={"exact_diff": 0},
                    )
            except Exception:
                pass

        # 4. Symbolic Equivalence via sp.simplify
        try:
            diff = sp.simplify(cand_expr - ref_expr)
            if diff == 0:
                return VerificationRecord(
                    problem_id=problem_id,
                    category=category,
                    original_problem=original_problem,
                    model_name=model_name,
                    model_raw_answer=model_output,
                    parsed_expression=str(cand_expr),
                    reference_answer=reference_str,
                    is_correct=True,
                    error_category=None,
                    latency_ms=latency_ms,
                    token_count=token_count,
                    verification_method="symbolic_diff",
                    details={"residual": 0},
                )
        except Exception:
            pass

        # 5. Numerical Tolerance Check
        try:
            cand_eval = complex(cand_expr.evalf())
            ref_eval = complex(ref_expr.evalf())

            if abs(ref_eval) < 1e-12:
                is_close = abs(cand_eval - ref_eval) < self.abs_tol
            else:
                rel_diff = abs(cand_eval - ref_eval) / abs(ref_eval)
                is_close = (rel_diff < self.rel_tol) or (abs(cand_eval - ref_eval) < self.abs_tol)

            if is_close:
                return VerificationRecord(
                    problem_id=problem_id,
                    category=category,
                    original_problem=original_problem,
                    model_name=model_name,
                    model_raw_answer=model_output,
                    parsed_expression=str(cand_expr),
                    reference_answer=reference_str,
                    is_correct=True,
                    error_category=None,
                    latency_ms=latency_ms,
                    token_count=token_count,
                    verification_method="numerical_tol",
                    details={"cand_eval": str(cand_eval), "ref_eval": str(ref_eval)},
                )
        except Exception:
            pass

        # Failed match
        return VerificationRecord(
            problem_id=problem_id,
            category=category,
            original_problem=original_problem,
            model_name=model_name,
            model_raw_answer=model_output,
            parsed_expression=str(cand_expr),
            reference_answer=reference_str,
            is_correct=False,
            error_category="WrongValue",
            latency_ms=latency_ms,
            token_count=token_count,
            verification_method="symbolic_diff",
            details={"candidate": str(cand_expr), "reference": str(ref_expr)},
        )

    def _check_equation_root(
        self,
        cand_expr: sp.Expr,
        eq_expr: Optional[sp.Expr],
        var_sym: sp.Symbol,
        metadata: dict,
    ) -> Tuple[bool, Optional[str], dict]:
        """Substitute candidate root into equation, check residual and domain singularities."""
        if eq_expr is None:
            return False, "WrongValue", {"error": "Equation not provided"}

        # Check domain singularity in metadata
        singularity_cand = metadata.get("extraneous_candidate")
        if singularity_cand:
            try:
                if sp.simplify(cand_expr - sp.sympify(singularity_cand)) == 0:
                    return False, "ExtraneousRoot", {"root": str(cand_expr), "domain_violation": True}
            except Exception:
                pass

        try:
            subbed = eq_expr.subs(var_sym, cand_expr)
            # Check for nan or division by zero
            if subbed in [sp.nan, sp.zoo]:
                return False, "DomainViolation", {"root": str(cand_expr), "residual": "nan"}

            residual = sp.simplify(subbed)
            if residual == 0:
                return True, None, {"residual": 0, "root": str(cand_expr)}

            res_val = abs(complex(residual.evalf()))
            if res_val < self.abs_tol:
                return True, None, {"residual": res_val, "root": str(cand_expr)}
        except Exception as e:
            return False, "WrongValue", {"error": f"Substitution failed: {e}"}

        return False, "WrongValue", {"residual": str(residual), "root": str(cand_expr)}

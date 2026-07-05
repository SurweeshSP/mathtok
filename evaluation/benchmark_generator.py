"""
Procedural SymPy Benchmark Generator
Generates 1000+ expressions across various mathematical domains for evaluation.
"""
import json
import random
from pathlib import Path
from sympy import *

x, y, z, t, n, k = symbols('x y z t n k')
a, b, c, alpha, beta = symbols('a b c alpha beta')

def generate_algebra(count=200):
    exprs = []
    for i in range(count):
        deg1 = random.randint(2, 5)
        deg2 = random.randint(2, 5)
        poly = expand((x + random.randint(1, 5)*y)**deg1 * (x - random.randint(1,5))**deg2)
        exprs.append(str(poly))
    return exprs

def generate_calculus(count=200):
    exprs = []
    funcs = [sin, cos, tan, exp, log, sqrt]
    for i in range(count):
        f = random.choice(funcs)(x**random.randint(1,3) + random.randint(1,5))
        if random.random() < 0.5:
            exprs.append(str(diff(f, x)))
        else:
            exprs.append(str(integrate(f, x)))
    return exprs

def generate_probability(count=100):
    exprs = []
    for i in range(count):
        mu = random.randint(0, 5)
        sigma = random.randint(1, 5)
        pdf = (1/(sigma*sqrt(2*pi))) * exp(-((x-mu)**2)/(2*sigma**2))
        exprs.append(str(pdf))
    return exprs

def generate_statistics(count=100):
    exprs = []
    for i in range(count):
        n_val = random.randint(10, 100)
        mean_expr = Sum(x, (x, 1, n_val)) / n_val
        exprs.append(str(mean_expr))
    return exprs

def generate_linear_algebra(count=100):
    exprs = []
    for i in range(count):
        dim = random.randint(2, 4)
        M = Matrix(dim, dim, lambda i, j: random.randint(-5, 5))
        exprs.append(str(M.det()))
    return exprs

def generate_optimization(count=100):
    exprs = []
    for i in range(count):
        f = (x - random.randint(1, 5))**2 + (y - random.randint(1, 5))**2
        exprs.append(str(f))
    return exprs

def generate_logic(count=100):
    exprs = []
    for i in range(count):
        p, q, r = symbols('p q r')
        expr = Implies(p & q, r | ~p)
        exprs.append(str(expr))
    return exprs

def generate_number_theory(count=100):
    exprs = []
    for i in range(count):
        val = random.randint(100, 1000)
        exprs.append(str(factorint(val)))
    return exprs

def generate_all():
    print("Generating Algebra...")
    alg = [{"category": "algebra", "expression": e} for e in generate_algebra(200)]
    print("Generating Calculus...")
    calc = [{"category": "calculus", "expression": e} for e in generate_calculus(200)]
    print("Generating Probability...")
    prob = [{"category": "probability", "expression": e} for e in generate_probability(100)]
    print("Generating Statistics...")
    stat = [{"category": "statistics", "expression": e} for e in generate_statistics(100)]
    print("Generating Linear Algebra...")
    linalg = [{"category": "linear_algebra", "expression": e} for e in generate_linear_algebra(100)]
    print("Generating Optimization...")
    opt = [{"category": "optimization", "expression": e} for e in generate_optimization(100)]
    print("Generating Logic...")
    logic = [{"category": "logic", "expression": e} for e in generate_logic(100)]
    print("Generating Number Theory...")
    nt = [{"category": "number_theory", "expression": e} for e in generate_number_theory(100)]
    
    all_data = alg + calc + prob + stat + linalg + opt + logic + nt
    # assign IDs
    for i, item in enumerate(all_data):
        item["id"] = i + 1
        item["latex"] = ""
        item["canonical"] = ""
        
    out_path = Path(__file__).parent / "datasets" / "benchmark_v2.0.json"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Generated {len(all_data)} expressions to {out_path}")

if __name__ == "__main__":
    generate_all()

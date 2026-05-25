"""
MathTok setup — installable as:  pip install -e .
"""
from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="mathtok",
    version="0.1.0",
    description=(
        "A Hybrid Canonicalized AST-Based Tokenization Framework "
        "for Mathematical Language Modeling"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Surweesh SP",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "notebooks*", "paper*"]),
    install_requires=[
        "sympy>=1.12",
        "antlr4-python3-runtime==4.11.1",
        "tokenizers>=0.15.0",
        "transformers>=4.38.0",
        "numpy>=1.26.0",
        "regex>=2023.12.25",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "eval": ["scipy>=1.12.0", "matplotlib>=3.8.0", "seaborn>=0.13.0", "networkx>=3.2"],
        "dev":  ["pytest>=8.0.0", "pytest-cov>=5.0.0", "jupyter>=1.0.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        "console_scripts": [
            "mathtok=mathtok.pipeline:cli",
        ]
    },
)

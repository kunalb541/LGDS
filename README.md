## Task-Relative Descriptive Privilege in Linear Gaussian Dynamical Systems

**Kunal Bhatia** — Independent Researcher, Heidelberg, Germany
ORCID: [0009-0007-4447-6325](https://orcid.org/0009-0007-4447-6325)

## The result in plain language

In a linear Gaussian dynamical system, if you want to compress the system state into
a rank-*r* linear observation, the optimal observer is uniquely determined by the top-*r*
eigenspace of a **target information matrix** — a symmetric PSD matrix encoding which
directions of the state space matter for the prediction target.

Two different prediction targets (different variables, different horizons) generally
have *different* optimal observers. No single rank-*r* observer can be simultaneously
optimal for both unless their information matrices share a common top-*r* eigenspace.
That shared-eigenspace condition is called **r-coherence**.

The main theorem: global privilege holds if and only if the target family is r-coherent.
Outside r-coherence, the optimal observer is task-relative.

The proof is two lines: the Ky Fan maximum principle gives the optimal subspace
exactly; two distinct subspaces cannot be spanned by the same matrix.

## All computations are exact — no simulation

Every number in the paper is computed from a closed-form Bayes risk formula:

```
R(U; C, τ) = tr(CΣCᵀ) - tr(U M_{C,τ} Uᵀ)
```

where `M_{C,τ} = Σ^{1/2} (Aᵀ)^τ CᵀC Aᵗ Σ^{1/2}` is the target information matrix.

There is no Monte Carlo simulation, no stochastic sampling, and no approximation.
The minimax observer (Table 1) is found by repeated projected optimisation on
the Grassmannian Gr(2, 8), which is an analytic optimisation problem.

## Reproducing the numerical illustrations

**Requirements:** Python ≥ 3.9, NumPy, SciPy.

```bash
# Activate your Python environment
source /path/to/your/venv/bin/activate

# Regenerate all paper numbers and macros
python lgds.py

# Compile the paper (requires TeX Live / MacTeX with latexmk)
bash build.sh
```

The script `lgds.py` writes `outputs/data/paper_macros.tex` containing all 14
numerical macros used in `paper.tex`. Every number in the paper is generated here.

**Tests:**

```bash
python -m pytest tests/ -v
```

## File structure

```
lgds.py                       # all computation; writes outputs/data/paper_macros.tex
paper.tex                     # manuscript
refs.bib                      # bibliography
build.sh                      # compiles paper (python lgds.py + latexmk)
paper.pdf                     # compiled paper
outputs/data/paper_macros.tex # auto-generated numerical macros
tests/test_lgds.py            # correctness tests
```

## Paper structure

- **Section 2** — System and observer family (LGDS, Grassmannian)
- **Section 3** — Theorem: Bayes risk formula → top-eigenspace optimality (Proposition 1)
  → no-global-privilege corollary → r-coherence definition and equivalence
- **Section 4** — Numerical illustrations: Condition A (no privilege, Γ = 0.026 > 0),
  Condition B (structural coherence, max angle 0.000°, perturbation-robust)
- **Section 5** — Scope (what is and is not earned)

## Citation

Bhatia, K. (2026). *Task-Relative Descriptive Privilege in Linear Gaussian
Dynamical Systems*. Preprint.

## Task-Dependent Optimal Linear Observers in Linear Gaussian Dynamical Systems

**Kunal Bhatia** — Independent Researcher, Heidelberg, Germany
ORCID: [0009-0007-4447-6325](https://orcid.org/0009-0007-4447-6325)

Submitted to *Systems & Control Letters* (Elsevier).

## The result in plain language

In a stationary linear Gaussian dynamical system, the optimal rank-*r* linear
observer for predicting a target *Cx*(*t*+τ) is the top-*r* eigenspace of the
**target information matrix**

```
M(C,τ) = Σ^(1/2) (A^τ)^T C^T C A^τ Σ^(1/2)
```

— a symmetric PSD matrix encoding which directions of the state space matter
for the prediction target and horizon.

Two targets generally have *different* optimal observers. No single rank-*r*
observer can be simultaneously optimal for both unless their information matrices
share a common top-*r* eigenspace. That shared-eigenspace condition is called
**r-coherence**.

The main theorem: a common optimal observer exists if and only if the target
family is r-coherent. Outside r-coherence, the optimal observer is
target-dependent.

The proof is two lines: the Ky Fan variational principle gives the optimal
subspace exactly; two distinct subspaces cannot be spanned by the same matrix.

## All computations are exact — no simulation

Every number in the paper is computed from a closed-form Bayes risk formula:

```
R(U; C, τ) = tr(CΣCᵀ) - tr(U M(C,τ) Uᵀ)
```

There is no Monte Carlo simulation, no stochastic sampling, and no approximation.
The minimax observer (Table 1) is found by repeated projected optimisation on
the Grassmannian Gr(2, 8), which is an analytic optimisation problem.

## Reproducing the numerical illustrations

**Requirements:** Python ≥ 3.9, NumPy, SciPy.

```bash
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
lgds.py                        # all computation; writes outputs/data/paper_macros.tex
paper.tex                      # manuscript (with author info)
paper.pdf                      # compiled manuscript
paper_blind.tex                # blind manuscript for double-blind review
paper_blind.pdf                # compiled blind manuscript
refs.bib                       # bibliography
build.sh                       # compiles paper (python lgds.py + latexmk)
cover_letter.txt               # cover letter for SCL submission
abstract_field.txt             # plain-text abstract for Editorial Manager
declaration_of_interest.txt    # declaration of no competing interests
outputs/data/paper_macros.tex  # auto-generated numerical macros
tests/test_lgds.py             # correctness tests
```

## Paper structure

- **Section 2** — System and observer family (stationary LGDS, Grassmannian)
- **Section 3** — Theorem: Bayes risk formula → top-eigenspace optimality (Proposition 1)
  → target-dependent observer selection (Corollary 2) → r-coherence definition
  and exact biconditional (Corollary 4)
- **Section 4** — Numerical illustrations: Condition A (target-dependent, Γ = 0.026 > 0),
  Condition B (r-coherent, max angle 0.000°, perturbation-robust)
- **Section 5** — Scope

## Citation

Bhatia, K. (2026). *Task-Dependent Optimal Linear Observers in Linear Gaussian
Dynamical Systems*. Submitted to Systems & Control Letters.

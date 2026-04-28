### Common Optimal Finite-Rank Projections in Linear Gaussian Dynamical Systems

**Kunal Bhatia** — Independent Researcher, Heidelberg, Germany
ORCID: [0009-0007-4447-6325](https://orcid.org/0009-0007-4447-6325)

## Two manuscripts in this repository

This repository contains two manuscripts. They are not independent submissions:
the first is a special case / subsection of the second.

| File | Title | Pages | Status |
| --- | --- | --- | --- |
| `paper_expanded.tex` | *Common Optimal Finite-Rank Projections: Coherence, Coverability, and Regret* | 11 | **Main formal candidate** |
| `paper_blind.tex` / `paper.tex` | *Task-Dependent Optimal Linear Projections in Linear Gaussian Dynamical Systems* | 4 | Short note; superseded by the expanded paper for submission purposes |

The short LGDS note is retained as a fallback / clean short proof reference
for the predictive part. The expanded paper subsumes its result as
Lemma 6 (Bayes risk derivation) plus Corollary 7 (LGDS predictive iff), and
adds the causal/coverability theorem, the fork theorem, and the weak-glue
surrogate with instance-dependent regret bound. Do not submit both as
independent papers.

## The expanded paper in plain language

For a finite family of positive semidefinite (PSD) target operators on **R**^*d*
and a fixed projection rank *r*, when does a single rank-*r* orthogonal
projection achieve the maximum simultaneously for every target? The answer
depends on the scoring rule:

- **Predictive trace** *S*(P) = tr(*P M_t*) — Ky Fan trace, the squared-error
  prediction loss in linear Gaussian settings. Common optimum exists iff the
  family is **r-coherent**: all *M_t* share a common top-*r* eigenspace.
- **Causal operator norm** *S*(P) = λ_max(*P K_t P*) — strongest accessible
  intervention response under projector *P*. Common optimum exists iff the
  family is **r-coverable**: a single rank-*r* subspace nontrivially intersects
  the leading invariant subspace of every *K_t*.

r-coherence implies r-coverability strictly: a *3 × 3* witness with
T1 = diag(5,4,0), T2 = diag(5,0,4) is 2-coverable (both share top eigenvector e1)
but not 2-coherent (top-2 eigenspaces are span{e1,e2} ≠ span{e1,e3}).

When coverability fails over a regime of local intervention Jacobians {*J_i*}
with output Grams *A_i = J_i J_i^T*, the eigenvalue-weighted surrogate
*B_λ = Σ_i w_i λ_1(A_i) u_i u_i^T* converts the non-Ky-Fan operator-norm
objective into a Ky Fan eigenproblem. The sandwich
*F̃(P) ≤ F(P) ≤ F̃(P) + Σ_i w_i λ_2(A_i)* and instance-dependent regret bound
*F(P_F\*) − F(P_B\*) ≤ Σ_i w_i λ_2(A_i)(1 − ‖P_F\* u_i‖²)* are tight in three
exact regimes (rank-1, strong glue, common-shape).

## The LGDS short note in plain language

In a stationary linear Gaussian dynamical system, the optimal rank-*r* linear
projection for predicting a target *Cx*(*t*+τ) is the top-*r* eigenspace of the
**target information matrix**

```
M(C,τ) = Σ^(1/2) (A^τ)^T C^T C A^τ Σ^(1/2)
```

— a symmetric PSD matrix encoding which directions of the state space matter
for the prediction target and horizon.

Two targets generally have *different* optimal projections. No single rank-*r*
projection can be simultaneously optimal for both unless their information
matrices share a common top-*r* eigenspace. That shared-eigenspace condition
is called **r-coherence**.

The main theorem: a common optimal projection exists if and only if the
target family is r-coherent. Outside r-coherence, the optimal projection
is target-dependent. Corollary 4 (the iff characterization) has an explicit
forward/backward proof: the forward direction is immediate; the backward
direction uses the strict-gap uniqueness of the top-*r* eigenspace
(or the dominant invariant subspace under degeneracy) to force all
information matrices to share a common dominant subspace — that step is
the non-trivial content.

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
lgds.py                        # all LGDS computation; writes outputs/data/paper_macros.tex
paper_expanded.tex             # MAIN — Common Optimal Finite-Rank Projections (11 pp.)
paper_expanded.pdf             # compiled main manuscript
paper.tex                      # short LGDS note (with author info, RevTeX format)
paper.pdf                      # compiled short note
paper_blind.tex                # blind short LGDS note (elsarticle, for SCL)
paper_blind.pdf                # compiled blind short note
refs.bib                       # bibliography (shared by all three)
build.sh                       # compiles short notes (python lgds.py + latexmk)
cover_letter.txt               # cover letter for SCL submission of short note
abstract_field.txt             # plain-text abstract for Editorial Manager
declaration_of_interest.txt    # declaration of no competing interests
outputs/data/paper_macros.tex  # auto-generated numerical macros (LGDS)
tests/test_lgds.py             # correctness tests for LGDS computations
```

## Paper structure (expanded paper, paper_expanded.tex)

- **Section 1** — Introduction: two scoring rules, four contributions, position
  vs Gramians/balanced truncation/sensor selection
- **Section 2** — Setup: Grassmannian Gr(*r,d*), predictive and causal scores,
  finite global privilege definition
- **Section 3** — Predictive global privilege via r-coherence (Theorem 3,
  for arbitrary PSD families) → LGDS instantiation (Lemma 6 + Corollary 7) →
  regret-angle bound (Corollary 8) → approximate predictive coherence
  (Corollary 9)
- **Section 4** — Causal global privilege via r-coverability: rank-1-driven
  local maximizer (Proposition 10) → r-coverability definition (Definition 11)
  → causal iff theorem (Theorem 12) → strong-glue corollary (Corollary 14)
- **Section 5** — The fork: r-coherence implies r-coverability strictly
  (Theorem 15) → predictive/causal dissociation (Remark 16)
- **Section 6** — Weak-glue surrogate: sandwich (Theorem 17) → coarse and
  instance-dependent regret bounds → three exact regimes (rank-1 / strong glue /
  common-shape; Corollary 19)
- **Section 7** — Numerical illustrations: LGDS predictive (Γ = 0.026 > 0),
  3×3 fork witness, 3×3 weak-glue counterexample (coarse 36×, instance-dep 6×)
- **Section 8** — Scope and discussion

## Paper structure (short LGDS note, paper.tex / paper_blind.tex)

- **Section 2** — System and projection family (stationary LGDS, Grassmannian)
- **Section 3** — Theorem: Bayes risk formula → top-eigenspace optimality
  (Proposition 1) → target-dependent projection selection (Corollary 2) →
  r-coherence definition and structural iff characterization with explicit
  proof (Corollary 4) → minimax shared projection and approximate coherence
  (§3.5)
- **Section 4** — Numerical illustrations: Condition A (target-dependent,
  Γ = 0.026 > 0), Condition B (r-coherent, max angle 0.000°)
- **Section 5** — Scope

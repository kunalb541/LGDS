### Coherence, Coverability, and Regret for Finite-Rank Projections of PSD Families

**Kunal Bhatia** — Independent Researcher, Heidelberg, Germany
ORCID: [0009-0007-4447-6325](https://orcid.org/0009-0007-4447-6325)

Target venue: *Linear Algebra and its Applications* (Elsevier).

## What the paper does

For a finite family of positive semidefinite (PSD) operators on **R**^*d*
and a fixed projection rank *r*, when does a single rank-*r* orthogonal
projection achieve the maximum simultaneously for every operator? The
answer depends on the scoring rule:

- **Trace score** *S*(P) = tr(*P M_t*) — Ky Fan trace.
  A common optimum exists iff the family is **r-coherent**: all *M_t*
  share a common dominant *r*-dimensional invariant subspace.
- **Operator-norm score** *S*(P) = λ_max(*P K_t P*) — Rayleigh quotient.
  A common optimum exists iff the family is **r-coverable**: a single
  rank-*r* subspace nontrivially intersects the leading eigenspace of
  every *K_t*.

**The fork.** *r*-coherence implies *r*-coverability strictly: a 3×3
witness with *T₁* = diag(5,4,0), *T₂* = diag(5,0,4) is 2-coverable
(both share top eigenvector *e₁*) but not 2-coherent (top-2 eigenspaces
span{*e₁*,*e₂*} ≠ span{*e₁*,*e₃*}).

**Weak-glue surrogate.** When coverability fails for a finite-support
regime {*A_i*} of local PSD operators, the eigenvalue-weighted surrogate

```
B_λ = Σ_i w_i λ_1(A_i) u_i u_i^T
```

(with *u_i* the leading eigenvector of *A_i*) converts the non-Ky–Fan
operator-norm objective into a Ky–Fan eigenproblem. The sandwich
*F̃(P) ≤ F(P) ≤ F̃(P) + Σ_i w_i λ_2(A_i)* and the instance-dependent
regret bound

```
F(P_F*) − F(P_B*) ≤ Σ_i w_i λ_2(A_i)(1 − ||P_F* u_i||²)
```

are tight in three regimes (rank-1 *A_i*, *r*-coverable case, common-shape
family).

**Stationary LGDS instance.** The trace-score theorem instantiates in
stationary linear Gaussian dynamical systems. Squared-error prediction
risk for target *Cx*(*t*+τ) is governed by the target information matrix

```
M(C,τ) = Σ^(1/2) (A^τ)^T C^T C A^τ Σ^(1/2)
```

and the *r*-coherence iff characterization gives Corollary 7.

## All computations are exact — no simulation

Every number in the paper is computed analytically:
- The LGDS predictive instance (Section 7.1) uses the closed-form Bayes
  risk *R(U;C,τ) = tr(CΣCᵀ) − tr(U M(C,τ) Uᵀ)* and exact
  eigendecomposition of *M(C,τ)*. The minimax projection
  *P_joint* requires Riemannian optimisation on Gr(2, 8), but only at
  the minimax level — single-target optima are closed-form.
- The fork witness (Section 7.2) and the weak-glue counterexample
  (Section 7.3) are stated as exact 3×3 matrices in equations (12) and
  (21); all reported quantities follow analytically.

There is no Monte Carlo simulation and no stochastic sampling.

## Reproducing the numerical illustrations

**Requirements:** Python ≥ 3.9, NumPy, SciPy.

```bash
# Regenerate all paper numbers and macros, then compile
bash build.sh
```

`build.sh` runs `python lgds.py` (which writes `outputs/data/paper_macros.tex`
containing all numerical macros) and then compiles `paper.tex` with
`latexmk`. Requires TeX Live / MacTeX with `latexmk`.

**Tests:**

```bash
python -m pytest tests/ -v
```

## File structure

```
lgds.py                        # numerical computation; writes outputs/data/paper_macros.tex
paper.tex                      # the manuscript (12 pages, elsarticle format)
paper.pdf                      # compiled manuscript
refs.bib                       # bibliography
build.sh                       # python lgds.py + latexmk
cover_letter.txt               # cover letter for LAA submission
abstract_field.txt             # plain-text abstract for Editorial Manager
declarationStatement.docx      # declaration of no competing interests
outputs/data/paper_macros.tex  # auto-generated numerical macros
tests/test_lgds.py             # correctness tests for the lgds.py computations
```

## Paper structure

- **§1 Introduction** — two scoring rules, four contributions, position
  vs CPC / joint-diagonalisation / Gramian / sensor-selection literature.
  Comparison table of the two existence laws.
- **§2 Setup** — Grassmannian Gr(*r,d*), trace and operator-norm scoring
  rules, common optimal rank-*r* projection (Definition 1), dominant
  invariant subspace and leading eigenspace (Definition 2).
- **§3 Trace-Score Common Optimality via *r*-Coherence** — Theorem 3
  for arbitrary PSD families; LGDS instantiation (Lemma 6 + Corollary 7);
  regret–angle bound (Corollary 8); approximate *r*-coherence
  (Corollary 9).
- **§4 Operator-Norm Common Optimality via *r*-Coverability** — local
  linearised intervention-response motivation (§4.1); rank-1-driven
  local maximizer (Proposition 10); *r*-coverability definition and
  testing remark; operator-norm theorem (Theorem 12); simple-top
  sufficient condition (Corollary 14).
- **§5 The Fork** — Theorem 15: *r*-coherence implies *r*-coverability
  strictly; 3×3 witness; trace vs operator-norm dissociation.
- **§6 Surrogate Approximation When Coverability Fails (weak glue)** —
  sandwich (Theorem 17); instance-dependent regret (Theorem 18); three
  exact regimes (Corollary 19); value-vs-subspace remark.
- **§7 Numerical Illustrations** — LGDS trace-score instance
  (Γ_tr = 0.026 > 0); 3×3 fork witness with explicit minimax derivation
  Γ_tr^fork = 2; 3×3 weak-glue counterexample (coarse bound 1.15 vs
  instance-dependent 0.197 vs actual regret 0.032).
- **§8 Scope and Discussion** — what is earned, what is not earned,
  practical implications, and the surrogate's computational value in
  high-dimensional regimes.

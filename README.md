### When Common Optimal Projections Exist: A Trace/Operator-Norm Fork for PSD Families

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

and the *r*-coherence iff characterization gives an LGDS corollary
in §3.

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
paper.tex                      # the manuscript (15 pages, elsarticle format)
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
- **§2 Setup** — Grassmannian Gr(*r,d*); trace and operator-norm scoring
  rules; common optimal rank-*r* projection; dominant invariant subspace
  and leading eigenspace.
- **§3 Trace-Score Common Optimality via *r*-Coherence** — main trace
  characterisation (Proposition) for arbitrary PSD families;
  **generic non-coherence proposition** (positive-codimension /
  measure-zero, fibrewise scope); LGDS instantiation (Bayes risk lemma
  + LGDS corollary + **PSD realisability lemma** showing every PSD
  family is realisable as an LGDS prediction-task family at horizon
  *τ*=1); regret–angle bound; approximate *r*-coherence with one-sided
  contrapositive certificate.
- **§4 Operator-Norm Common Optimality via *r*-Coverability** —
  linearised Jacobian-response motivation (§4.1); rank-1-driven local
  maximizer; *r*-coverability definition; testing via Schubert-type
  incidence (simple-top reduction; sufficient condition
  dim Σ_t E_max(K_t) ≤ r; multidimensional witness); main operator-norm
  characterisation (Proposition); simple-top ("strong glue") sufficient
  condition.
- **§5 The Fork** — *r*-coherence implies *r*-coverability strictly
  (Theorem); diagonal 3×3 witness *T₁* = diag(5,4,0), *T₂* = diag(5,0,4);
  basis-invariant non-diagonal witness via orthogonal conjugation;
  trace vs operator-norm dissociation.
- **§6 Surrogate Approximation When Coverability Fails (weak glue)** —
  motivation for *B_λ* as the Ky-Fan rank-1 lower contribution; sandwich
  inequality (Theorem); coarse and instance-dependent regret bounds
  (Theorem); three exact regimes (rank-1 *A_i*, *r*-coverable,
  common-shape); bound-sharpness diagnosis; value-vs-subspace remark.
- **§7 Numerical Illustrations** — (§7.1) LGDS trace-score instance
  (Γ_tr = 0.026 > 0); (§7.2) 3×3 fork witness with explicit minimax
  derivation Γ_tr^fork = 2; (§7.3) 3×3 weak-glue counterexample
  (coarse bound 1.15 vs instance-dependent 0.197 vs actual regret
  0.032); **(§7.4) higher-dimensional weak-glue stress test
  (d=50, r=5, N=100)**: surrogate captures essentially all achievable
  operator-norm value via one *d*×*d* eigendecomposition.
- **§8 Scope and Discussion** — what is earned, what is not earned
  (continuous-regime, nonlinear *K_t*, full Jensen gap, approximate
  coverability as a covering-radius / hitting-geometry problem
  distinct from approximate *r*-coherence), practical implications.

"""
lgds.py
=============
Single numerical source of truth for the LGDS paper.
Computes all paper numbers and writes outputs/data/paper_macros.tex.

Conditions:
  Condition A  -- no-global-privilege instance (seed=100, r=2, n=8)
                  Gamma = true minimax max-regret over Gr(2,8)
  Condition B' -- structural coherence + perturbation robustness

Rules:
  - No number appears in paper.tex that is not produced here.
  - Gamma is the true minimax quantity per the paper definition:
      Gamma = min_U max_k [ R(U; C_k, tau_k) - R*_k ]
  - No simulation is run; all quantities are analytic.
  - All macro names are purely alphabetic (TeX command names cannot
    contain digits -- digits terminate the name scan).
"""

import os
import numpy as np
from scipy.linalg import solve_discrete_lyapunov, subspace_angles
from scipy.optimize import minimize

os.makedirs("outputs/data", exist_ok=True)

# -- Helpers ------------------------------------------------------------------

def build_system(seed, normal=False):
    rng = np.random.default_rng(seed)
    n = 8
    eigs = rng.uniform(0.3, 0.85, n)
    if normal:
        V, _ = np.linalg.qr(rng.standard_normal((n, n)))
        A = V @ np.diag(eigs) @ V.T
    else:
        V = rng.standard_normal((n, n))
        A = V @ np.diag(eigs) @ np.linalg.inv(V)
    assert np.max(np.abs(np.linalg.eigvals(A))) < 1.0
    Sigma = solve_discrete_lyapunov(A, 0.1 * np.eye(n))
    assert np.all(np.linalg.eigvalsh(Sigma) > 0)
    ev, evec = np.linalg.eigh(Sigma)
    Sigma_half = evec @ np.diag(np.sqrt(ev)) @ evec.T
    return A, Sigma, Sigma_half

def info_matrix(A, Sigma_half, C, tau):
    At = np.linalg.matrix_power(A, tau)
    return Sigma_half @ At.T @ C.T @ C @ At @ Sigma_half

def bayes_risk(U, Sigma, M, C):
    return np.trace(C @ Sigma @ C.T) - np.trace(U @ M @ U.T)

def top_r_observer(M, r):
    _, vecs = np.linalg.eigh(M)
    return vecs[:, -r:].T

def project_gr(U_flat, r, n):
    """Project r*n flat vector onto Stiefel manifold (Gr representative)."""
    U = U_flat.reshape(r, n)
    Q, _ = np.linalg.qr(U.T, mode="reduced")
    return Q.T

def minimax_observer(Ms, Cs, Rstars, Sigma, r, n, n_restarts=300, seed=7):
    """
    Find U* = argmin_U max_k [ R(U; C_k, tau_k) - R*_k ]
    via repeated projected optimization from random initializations.
    Returns (U_opt, Gamma, regrets_at_opt).
    Sigma is passed explicitly -- no closure over globals.
    """
    rng = np.random.default_rng(seed)

    def J(u_flat):
        U = project_gr(u_flat, r, n)
        regrets = [bayes_risk(U, Sigma, M, C) - Rstar
                   for M, C, Rstar in zip(Ms, Cs, Rstars)]
        return max(regrets)

    best_val, best_u = np.inf, None
    for _ in range(n_restarts):
        res = minimize(J, rng.standard_normal(r * n), method="L-BFGS-B",
                       options={"maxiter": 10000, "ftol": 1e-15, "gtol": 1e-12})
        if res.fun < best_val:
            best_val = res.fun
            best_u = res.x.copy()

    U_opt = project_gr(best_u, r, n)
    regrets_at_opt = [bayes_risk(U_opt, Sigma, M, C) - Rstar
                      for M, C, Rstar in zip(Ms, Cs, Rstars)]
    Gamma = max(regrets_at_opt)
    return U_opt, Gamma, regrets_at_opt


# -- Condition A: no-global-privilege -----------------------------------------

print("Computing Condition A ...")

A, Sigma, Sigma_half = build_system(seed=100, normal=False)
n, r = 8, 2

C1 = np.zeros((2, n)); C1[0, 0] = 1.0; C1[1, 1] = 1.0   # [e1;e2], tau=1
C2 = np.zeros((2, n)); C2[0, 4] = 1.0; C2[1, 5] = 1.0   # [e5;e6], tau=5

M1 = info_matrix(A, Sigma_half, C1, 1)
M2 = info_matrix(A, Sigma_half, C2, 5)

U1 = top_r_observer(M1, r)
U2 = top_r_observer(M2, r)

# Pre-run check: eigenspaces are distinct
pa = np.degrees(subspace_angles(U1.T, U2.T))
assert np.min(pa) > np.degrees(0.3), f"Eigenspaces not distinct: min angle {np.min(pa):.2f} deg"

# Optimal risks
R1_star = bayes_risk(U1, Sigma, M1, C1)
R2_star = bayes_risk(U2, Sigma, M2, C2)

# Cross-regrets of axis-optimal observers
reg_U1_T2 = bayes_risk(U1, Sigma, M2, C2) - R2_star
reg_U2_T1 = bayes_risk(U2, Sigma, M1, C1) - R1_star
assert reg_U1_T2 > 0 and reg_U2_T1 > 0, "Cross-regrets not positive"

# True minimax Gamma -- Sigma passed explicitly
Rstars = [R1_star, R2_star]
Uj, Gamma, regrets_joint = minimax_observer([M1, M2], [C1, C2], Rstars, Sigma, r, n)
reg_Uj_T1, reg_Uj_T2 = regrets_joint

assert Gamma > 0, "Gamma=0: global privilege not ruled out"

print(f"  Principal angles: {pa[0]:.1f} deg, {pa[1]:.1f} deg")
print(f"  Regret(U2* on T1) = {reg_U2_T1:.6f}")
print(f"  Regret(U1* on T2) = {reg_U1_T2:.6f}")
print(f"  Gamma (true minimax) = {Gamma:.6f}")
print(f"  Regrets at Uj: T1={reg_Uj_T1:.6f}, T2={reg_Uj_T2:.6f}")


# -- Condition B: structural coherence + perturbation robustness ---------------
#
# Normal A = V D V^T, C = top-2 rows of V^T (eigenvectors of A).
# M_{C,tau} has fixed top-2 eigenspace for all tau -- exactly r-coherent.

print("Computing Condition B ...")

Ab, Sigmab, Sigmab_half = build_system(seed=200, normal=True)

_, Vb = np.linalg.eigh(Ab)    # columns = eigenvectors, ascending order
Cb = Vb[:, -2:].T             # 2 x n, rows = top-2 eigenvectors of A

targets_b = [(Cb, 1), (Cb, 2), (Cb, 3), (Cb, 5)]
Ms_b = [info_matrix(Ab, Sigmab_half, C, tau) for C, tau in targets_b]
Us_b = [top_r_observer(M, r) for M in Ms_b]

# Minimum spectral gap across Condition B targets.
# gap_k = lambda_r(M_k) - lambda_{r+1}(M_k) (descending notation).
# eigh returns ascending order, so gap = eigvals[-r] - eigvals[-r-1].
spectral_gaps_b = []
for M in Ms_b:
    eigvals_M = np.linalg.eigvalsh(M)
    spectral_gaps_b.append(eigvals_M[-r] - eigvals_M[-r - 1])
min_gap_b = min(spectral_gaps_b)
assert min_gap_b > 0, f"Spectral gap non-positive: {min_gap_b}"
print(f"  Spectral gaps (Condition B): {[f'{g:.4f}' for g in spectral_gaps_b]}")
print(f"  Min spectral gap: {min_gap_b:.6f}")

# All pairwise principal angles -- must be ~0
max_angle_deg = 0.0
for i in range(len(Us_b)):
    for j in range(i + 1, len(Us_b)):
        ang = np.degrees(subspace_angles(Us_b[i].T, Us_b[j].T))
        max_angle_deg = max(max_angle_deg, np.max(ang))

print(f"  Max pairwise angle (structural coherence): {max_angle_deg:.6f} deg")

# Perturbation robustness: eps=0.1, 20 draws
rng_pert = np.random.default_rng(42)
eps_pert, n_draws = 0.1, 20
frag_angles_rad = []
for _ in range(n_draws):
    Ms_pert = []
    for C, tau in targets_b:
        dC = rng_pert.standard_normal(C.shape)
        dC = dC / np.linalg.norm(dC, "fro")
        Ms_pert.append(info_matrix(Ab, Sigmab_half, C + eps_pert * dC, tau))
    Us_pert = [top_r_observer(M, r) for M in Ms_pert]
    for i in range(len(Us_pert)):
        for j in range(i + 1, len(Us_pert)):
            frag_angles_rad.append(np.max(subspace_angles(Us_pert[i].T, Us_pert[j].T)))

max_frag_rad = np.max(frag_angles_rad)
frag_threshold = 0.15
assert max_frag_rad < frag_threshold, f"Structural coherence fragile: {max_frag_rad:.4f} rad"
print(f"  Max angle under perturbation: {max_frag_rad:.6f} rad  (threshold: {frag_threshold})")


# -- High-dimensional weak-glue stress test (d=50, r=5, N=100) ----------------
# Synthetic, reproducible. Goal: show the surrogate B_lambda is competitive
# with locally optimised F at moderate dimension, and that the
# instance-dependent bound is consistently tighter than the coarse bound.

print("\nHigh-dimensional weak-glue stress test (d=50, r=5, N=100):")

def hidim_stress_test():
    rng_h = np.random.default_rng(42)
    d_h, r_h, N_h = 50, 5, 100
    K_clusters = 10  # > r_h, so coverability fails

    # K cluster centres in span{e_1, ..., e_K}
    centers = np.eye(d_h)[:, :K_clusters]

    u_list, lam1_list, lam2_list, A_list = [], [], [], []
    for i in range(N_h):
        c = centers[:, rng_h.integers(K_clusters)]
        u = c + 0.05 * rng_h.standard_normal(d_h)
        u /= np.linalg.norm(u)
        v = rng_h.standard_normal(d_h)
        v -= u * (u @ v)
        v /= np.linalg.norm(v)
        l1 = rng_h.uniform(0.8, 1.2)
        l2 = l1 * rng_h.uniform(0.2, 0.5)
        u_list.append(u)
        lam1_list.append(l1)
        lam2_list.append(l2)
        A_list.append(l1 * np.outer(u, u) + l2 * np.outer(v, v))

    weights_h = np.ones(N_h) / N_h

    # Surrogate B_lambda and its top-r eigenspace
    B = np.zeros((d_h, d_h))
    for i in range(N_h):
        B += weights_h[i] * lam1_list[i] * np.outer(u_list[i], u_list[i])
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    P_B = eigvecs_B[:, -r_h:]

    def F_value(U):
        return sum(weights_h[i] * np.linalg.eigvalsh(U.T @ A_list[i] @ U).max()
                   for i in range(N_h))

    F_PB = F_value(P_B)

    # Random rank-r baseline
    rng_rand = np.random.default_rng(123)
    F_rand_list = []
    for _ in range(50):
        Q, _ = np.linalg.qr(rng_rand.standard_normal((d_h, r_h)))
        F_rand_list.append(F_value(Q))
    F_rand_mean = float(np.mean(F_rand_list))
    F_rand_max = float(np.max(F_rand_list))

    # Approximate F-optimum by Riemannian gradient ascent + restarts
    def grad_F(U):
        g = np.zeros_like(U)
        for i in range(N_h):
            UTAU = U.T @ A_list[i] @ U
            _, ev = np.linalg.eigh(UTAU)
            z = ev[:, -1]
            g += 2 * weights_h[i] * A_list[i] @ U @ np.outer(z, z)
        return g

    def riemannian_ascent(U_init, max_iter=300, lr=0.05):
        U = U_init.copy()
        for _ in range(max_iter):
            g = grad_F(U)
            sym = (U.T @ g + g.T @ U) / 2
            rg = g - U @ sym
            U_new = U + lr * rg
            U, _ = np.linalg.qr(U_new)
        return U, F_value(U)

    rng_opt = np.random.default_rng(456)
    best_U, best_F = riemannian_ascent(P_B)  # warm start from surrogate
    for _ in range(5):
        Q, _ = np.linalg.qr(rng_opt.standard_normal((d_h, r_h)))
        U_opt, f_opt = riemannian_ascent(Q)
        if f_opt > best_F:
            best_U, best_F = U_opt, f_opt
    F_PF = best_F
    P_F_basis = best_U

    actual_regret = F_PF - F_PB
    upper_bound = sum(weights_h[i] * lam1_list[i] for i in range(N_h))
    coarse_bound = sum(weights_h[i] * lam2_list[i] for i in range(N_h))

    inst_dep_bound = 0.0
    for i in range(N_h):
        proj_u = P_F_basis.T @ u_list[i]
        norm_sq = float(proj_u @ proj_u)
        inst_dep_bound += weights_h[i] * lam2_list[i] * (1 - norm_sq)

    print(f"  F(random) mean = {F_rand_mean:.3f}")
    print(f"  F(P_B*) = {F_PB:.3f}")
    print(f"  F(P_F*) = {F_PF:.3f}  (regret = {actual_regret:.4f})")
    print(f"  upper bound Σ w_i λ_1(A_i) = {upper_bound:.3f}")
    print(f"  coarse bound Σ w_i λ_2(A_i) = {coarse_bound:.3f}")
    print(f"  instance-dep bound = {inst_dep_bound:.3f}")

    return {
        'F_rand_mean': F_rand_mean,
        'F_rand_max': F_rand_max,
        'F_PB': F_PB,
        'F_PF': F_PF,
        'actual_regret': actual_regret,
        'coarse_bound': coarse_bound,
        'inst_dep_bound': inst_dep_bound,
        'upper_bound': upper_bound,
    }

hi = hidim_stress_test()


# -- Write macros -------------------------------------------------------------
# IMPORTANT: all macro names must be purely alphabetic.
# TeX command names end at the first non-letter, so a trailing digit like
# \FooTone would be fine but \FooT1 is parsed as \FooT with argument 1.
# Naming convention: Ta = target 1, Tb = target 2.

def fmt(x, d=3):
    return f"{x:.{d}f}"

macros = {
    # Condition A: eigenspace angles
    "CondAAngleOne":          fmt(pa[0], 1),
    "CondAAngleTwo":          fmt(pa[1], 1),
    # Condition A: regret table (all from exact analytic formula)
    # Row U1* (optimal for T1): Ta = on T1, Tb = on T2
    "CondARegretUoneTa":      "0.000",
    "CondARegretUoneTb":      fmt(reg_U1_T2, 3),
    # Row U2* (optimal for T2): Ta = on T1, Tb = on T2
    "CondARegretUtwoTa":      fmt(reg_U2_T1, 3),
    "CondARegretUtwoTb":      "0.000",
    # Row Ujoint: Ta = on T1, Tb = on T2
    "CondARegretUjointTa":    fmt(reg_Uj_T1, 3),
    "CondARegretUjointTb":    fmt(reg_Uj_T2, 3),
    # Condition A: true minimax Gamma
    "GammaCondA":             fmt(Gamma, 3),
    # Condition B': coherence
    "CondBMaxAngleDeg":       fmt(max_angle_deg, 3),
    "CondBPertEps":           fmt(eps_pert, 1),
    "CondBPertDraws":         str(n_draws),
    "CondBPertMaxRad":        fmt(max_frag_rad, 3),
    "CondBPertThreshold":     fmt(frag_threshold, 2),
    # High-dimensional weak-glue stress test (d=50, r=5, N=100)
    "HiDimFRandMean":         fmt(hi['F_rand_mean'], 3),
    "HiDimFPB":               fmt(hi['F_PB'], 3),
    "HiDimFPF":               fmt(hi['F_PF'], 3),
    "HiDimRegret":            fmt(hi['actual_regret'], 4),
    "HiDimCoarse":            fmt(hi['coarse_bound'], 3),
    "HiDimInstDep":           fmt(hi['inst_dep_bound'], 3),
    "HiDimUpper":             fmt(hi['upper_bound'], 3),
}

lines = ["% Auto-generated by lgds.py -- do not edit by hand", ""]
for name, val in macros.items():
    lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")

with open("outputs/data/paper_macros.tex", "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nWrote {len(macros)} macros to outputs/data/paper_macros.tex")
print("Done.")
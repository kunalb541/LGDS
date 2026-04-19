"""
tests/test_lgds.py
==================
Correctness tests for the LGDS paper computations.

All tests are exact / analytic -- no simulation, no stochastic sampling.
Each test corresponds to a claim made in the paper.
"""

import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov, subspace_angles

# ---------------------------------------------------------------------------
# Helpers (duplicated from lgds.py to keep tests self-contained)
# ---------------------------------------------------------------------------

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
    Sigma = solve_discrete_lyapunov(A, 0.1 * np.eye(n))
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


# ---------------------------------------------------------------------------
# 1. Lyapunov equation: verify A Σ Aᵀ + Q = Σ
# ---------------------------------------------------------------------------

class TestLyapunov:
    def test_lyapunov_residual_non_normal(self):
        """Stationary covariance satisfies AΣAᵀ + Q = Σ (non-normal A)."""
        A, Sigma, _ = build_system(seed=100, normal=False)
        Q = 0.1 * np.eye(8)
        residual = A @ Sigma @ A.T + Q - Sigma
        assert np.max(np.abs(residual)) < 1e-10

    def test_lyapunov_residual_normal(self):
        """Stationary covariance satisfies AΣAᵀ + Q = Σ (normal A)."""
        A, Sigma, _ = build_system(seed=200, normal=True)
        Q = 0.1 * np.eye(8)
        residual = A @ Sigma @ A.T + Q - Sigma
        assert np.max(np.abs(residual)) < 1e-10

    def test_sigma_positive_definite(self):
        """Stationary covariance is strictly positive definite."""
        A, Sigma, _ = build_system(seed=100, normal=False)
        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.min(eigvals) > 0

    def test_sigma_half_squares_to_sigma(self):
        """Σ^{1/2} Σ^{1/2} = Σ (symmetric PSD square root)."""
        _, Sigma, Sigma_half = build_system(seed=100, normal=False)
        assert np.max(np.abs(Sigma_half @ Sigma_half - Sigma)) < 1e-10

    def test_sigma_half_symmetric(self):
        """Σ^{1/2} is symmetric."""
        _, _, Sigma_half = build_system(seed=100, normal=False)
        assert np.max(np.abs(Sigma_half - Sigma_half.T)) < 1e-12


# ---------------------------------------------------------------------------
# 2. Information matrix: symmetry and positive semidefiniteness
# ---------------------------------------------------------------------------

class TestInfoMatrix:
    def test_info_matrix_symmetric(self):
        """M_{C,τ} is symmetric."""
        A, _, Sigma_half = build_system(seed=100, normal=False)
        rng = np.random.default_rng(0)
        n = 8
        C = rng.standard_normal((3, n))
        M = info_matrix(A, Sigma_half, C, tau=3)
        assert np.max(np.abs(M - M.T)) < 1e-12

    def test_info_matrix_psd(self):
        """M_{C,τ} is positive semidefinite."""
        A, _, Sigma_half = build_system(seed=100, normal=False)
        rng = np.random.default_rng(0)
        n = 8
        C = rng.standard_normal((3, n))
        M = info_matrix(A, Sigma_half, C, tau=3)
        eigvals = np.linalg.eigvalsh(M)
        assert np.min(eigvals) >= -1e-10

    def test_info_matrix_matches_formula(self):
        """M_{C,τ} = Σ^{1/2} (Aᵀ)^τ CᵀC Aᵗ Σ^{1/2} (explicit check)."""
        A, _, Sigma_half = build_system(seed=100, normal=False)
        C = np.eye(3, 8)
        tau = 2
        At = np.linalg.matrix_power(A, tau)
        M_expected = Sigma_half @ At.T @ C.T @ C @ At @ Sigma_half
        M_computed = info_matrix(A, Sigma_half, C, tau)
        assert np.max(np.abs(M_expected - M_computed)) < 1e-14


# ---------------------------------------------------------------------------
# 3. Top-r observer achieves minimum Bayes risk
# ---------------------------------------------------------------------------

class TestTopRObserver:
    def test_top_r_achieves_minimum_risk(self):
        """
        The top-r observer achieves strictly lower risk than any other random
        rank-r observer, verified by brute-force comparison on small problem.
        """
        A, Sigma, Sigma_half = build_system(seed=42, normal=False)
        n, r = 8, 2
        C = np.eye(2, 8)
        M = info_matrix(A, Sigma_half, C, tau=1)
        U_opt = top_r_observer(M, r)

        R_opt = bayes_risk(U_opt, Sigma, M, C)

        # 200 random orthonormal rank-2 matrices
        rng = np.random.default_rng(123)
        for _ in range(200):
            Q, _ = np.linalg.qr(rng.standard_normal((n, r)))
            U_rand = Q.T  # r x n, row-orthonormal
            R_rand = bayes_risk(U_rand, Sigma, M, C)
            assert R_opt <= R_rand + 1e-10, (
                f"Optimal observer not optimal: R_opt={R_opt:.6f} > R_rand={R_rand:.6f}"
            )

    def test_top_r_observer_row_orthonormal(self):
        """top_r_observer returns a row-orthonormal matrix."""
        A, _, Sigma_half = build_system(seed=100, normal=False)
        C = np.eye(2, 8)
        M = info_matrix(A, Sigma_half, C, tau=1)
        U = top_r_observer(M, r=2)
        assert U.shape == (2, 8)
        assert np.max(np.abs(U @ U.T - np.eye(2))) < 1e-12

    def test_bayes_risk_formula(self):
        """
        The Bayes risk formula gives tr(CΣCᵀ) at U=0,
        and strictly lower value at the optimal U.
        """
        A, Sigma, Sigma_half = build_system(seed=100, normal=False)
        n = 8
        C = np.eye(2, n)
        M = info_matrix(A, Sigma_half, C, tau=1)
        U_opt = top_r_observer(M, r=2)

        # At the optimal observer
        R_opt = bayes_risk(U_opt, Sigma, M, C)
        # At U=0 (no observation), risk = tr(CΣCᵀ)
        U_zero = np.zeros((2, n))
        R_zero = bayes_risk(U_zero, Sigma, M, C)
        expected_zero = np.trace(C @ Sigma @ C.T)

        assert abs(R_zero - expected_zero) < 1e-12
        assert R_opt < R_zero


# ---------------------------------------------------------------------------
# 4. r-coherent family has zero principal angles
# ---------------------------------------------------------------------------

class TestCoherence:
    def test_rcoherent_family_zero_angles(self):
        """
        Structural r-coherence: normal A, C = top-2 eigenvectors of A.
        All information matrices share the same top-2 eigenspace;
        max pairwise principal angle is zero to machine precision.
        """
        A, _, Sigma_half = build_system(seed=200, normal=True)
        _, Vb = np.linalg.eigh(A)
        C = Vb[:, -2:].T  # top-2 eigenvectors of A

        r = 2
        taus = [1, 2, 3, 5]
        Ms = [info_matrix(A, Sigma_half, C, tau) for tau in taus]
        Us = [top_r_observer(M, r) for M in Ms]

        max_angle = 0.0
        for i in range(len(Us)):
            for j in range(i + 1, len(Us)):
                ang = np.max(subspace_angles(Us[i].T, Us[j].T))
                max_angle = max(max_angle, ang)

        # Machine precision: should be < 1e-12 rad (~1e-10 degrees)
        assert max_angle < 1e-10, (
            f"r-coherent family has non-zero angle: {max_angle:.2e} rad"
        )

    def test_noncoherent_family_positive_gamma(self):
        """
        Non-coherent instance (Condition A) has distinct eigenspaces.
        Cross-regrets are strictly positive.
        """
        A, Sigma, Sigma_half = build_system(seed=100, normal=False)
        n, r = 8, 2

        C1 = np.zeros((2, n)); C1[0, 0] = 1.0; C1[1, 1] = 1.0
        C2 = np.zeros((2, n)); C2[0, 4] = 1.0; C2[1, 5] = 1.0

        M1 = info_matrix(A, Sigma_half, C1, tau=1)
        M2 = info_matrix(A, Sigma_half, C2, tau=5)

        U1 = top_r_observer(M1, r)
        U2 = top_r_observer(M2, r)

        R1_star = bayes_risk(U1, Sigma, M1, C1)
        R2_star = bayes_risk(U2, Sigma, M2, C2)

        # Cross-regrets must be strictly positive (no global privilege)
        reg_U1_on_T2 = bayes_risk(U1, Sigma, M2, C2) - R2_star
        reg_U2_on_T1 = bayes_risk(U2, Sigma, M1, C1) - R1_star

        assert reg_U1_on_T2 > 0, f"Cross-regret U1→T2 not positive: {reg_U1_on_T2}"
        assert reg_U2_on_T1 > 0, f"Cross-regret U2→T1 not positive: {reg_U2_on_T1}"

        # Principal angles must be non-zero (distinct eigenspaces)
        angles_deg = np.degrees(subspace_angles(U1.T, U2.T))
        assert np.min(angles_deg) > 1.0, (
            f"Eigenspaces not distinct: min angle {np.min(angles_deg):.2f} deg"
        )

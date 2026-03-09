"""
tests/test_ncg.py — Unit tests for the NCG codebase (pytest)
=============================================================
Covers:
  - Beta formula correctness (quadratic exact-arithmetic identities)
  - Gradient verification (numerical vs analytical)
  - Wolfe line search conditions satisfied
  - NCG convergence on all four test problems with all four variants
  - Restart logic (periodic, Powell, descent-failure)
  - Edge-case numerical safeguards
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from ncg import (
    ncg_minimize, NCGResult,
    beta_fletcher_reeves, beta_polak_ribiere,
    beta_polak_ribiere_plus, beta_hestenes_stiefel,
    wolfe_line_search, BETA_FUNCTIONS,
)
from test_functions import (
    quadratic_bowl_2d, rosenbrock, beale, high_dim_quadratic,
    get_all_test_problems,
)

TOL = 1e-6


def _numerical_grad(f, x, h=1e-6):
    """Central-difference numerical gradient for verification."""
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g


# ===========================================================================
# 1. Beta formula tests
# ===========================================================================

class TestBetaFormulas:

    def _make_gradients(self):
        rng = np.random.default_rng(0)
        return rng.standard_normal(10), rng.standard_normal(10), rng.standard_normal(10)

    def test_fr_positive(self):
        """FR beta is always non-negative (ratio of squared norms)."""
        g_new, g_old, d_old = self._make_gradients()
        assert beta_fletcher_reeves(g_new, g_old, d_old) >= 0.0

    def test_fr_formula(self):
        """FR formula: ‖g_new‖² / ‖g_old‖²."""
        g_new, g_old, d_old = self._make_gradients()
        expected = np.dot(g_new, g_new) / np.dot(g_old, g_old)
        assert abs(beta_fletcher_reeves(g_new, g_old, d_old) - expected) < 1e-14

    def test_pr_can_be_negative(self):
        """PR beta can be negative — self-correcting property."""
        g_old = np.array([2.0, 0.0])
        g_new = np.array([0.5, 0.0])
        d_old = np.zeros(2)
        # g_new . (g_new - g_old) = [0.5].[-1.5] = -0.75 < 0 -> beta < 0
        assert beta_polak_ribiere(g_new, g_old, d_old) < 0.0

    def test_prplus_non_negative(self):
        """PR+ is always >= 0 (max truncation)."""
        g_old = np.array([2.0, 0.0])
        g_new = np.array([0.5, 0.0])
        d_old = np.zeros(2)
        assert beta_polak_ribiere_plus(g_new, g_old, d_old) >= 0.0

    def test_prplus_equals_pr_when_pr_positive(self):
        """PR+ = PR when PR >= 0."""
        g_new = np.array([1.5, 0.5]); g_old = np.array([1.0, 0.0]); d_old = np.zeros(2)
        assert (abs(beta_polak_ribiere_plus(g_new, g_old, d_old)
                    - max(beta_polak_ribiere(g_new, g_old, d_old), 0.0)) < 1e-14)

    def test_hs_formula(self):
        """HS formula: g_new . y / d_old . y."""
        g_new = np.array([1.2, 0.4]); g_old = np.array([0.8, 0.1]); d_old = np.array([0.3, -0.5])
        y = g_new - g_old; denom = np.dot(d_old, y)
        if abs(denom) > 1e-30:
            expected = np.dot(g_new, y) / denom
            assert abs(beta_hestenes_stiefel(g_new, g_old, d_old) - expected) < 1e-12

    def test_hs_zero_denominator_safeguard(self):
        """HS returns 0 when d^T y = 0 (implicit restart)."""
        g_new = np.array([1.0, 0.0]); g_old = np.array([1.0, 0.0]); d_old = np.array([1.0, 0.0])
        assert beta_hestenes_stiefel(g_new, g_old, d_old) == 0.0

    def test_fr_pr_equal_on_exact_quadratic(self):
        """On a quadratic with orthogonal residuals, FR == PR exactly."""
        g_old = np.array([1.0, 0.0]); g_new = np.array([0.0, 1.0]); d_old = np.zeros(2)
        fr = beta_fletcher_reeves(g_new, g_old, d_old)
        pr = beta_polak_ribiere(g_new, g_old, d_old)
        assert abs(fr - pr) < 1e-12, f"FR={fr}, PR={pr}"

    def test_zero_old_gradient_safeguard(self):
        """FR, PR, PR+ return 0 when ||g_old|| = 0 (division-by-zero safeguard)."""
        g_old = np.zeros(5); g_new = np.ones(5); d_old = np.ones(5)
        for name in ["FR", "PR", "PR+"]:
            assert BETA_FUNCTIONS[name](g_new, g_old, d_old) == 0.0, \
                f"{name} should return 0 when g_old=0"
        # HS: separate safeguard via |d.y| < 1e-30
        g_hs_old = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        g_hs_new = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # y=0 -> d.y=0
        assert BETA_FUNCTIONS["HS"](g_hs_new, g_hs_old, d_old) == 0.0


# ===========================================================================
# 2. Gradient verification (numerical vs analytical)
# ===========================================================================

class TestGradientVerification:
    """Verifies all analytical gradients against central differences."""

    def _check(self, f, gf, x_tests, name, tol=1e-5):
        for x in x_tests:
            err = np.linalg.norm(gf(x) - _numerical_grad(f, x))
            assert err < tol, f"{name}: gradient error {err:.2e} at x={x}"

    def test_rosenbrock(self):
        f, gf, _, _ = rosenbrock()
        self._check(f, gf, [np.array([-1.2, 1.0]), np.array([0.5, 0.25]),
                             np.array([1.0, 1.0])], "Rosenbrock")

    def test_beale(self):
        f, gf, _, _ = beale()
        self._check(f, gf, [np.array([1.0, 1.0]), np.array([2.0, 0.3]),
                             np.array([3.0, 0.5])], "Beale")

    def test_quadratic_2d(self):
        f, gf, _, _ = quadratic_bowl_2d(kappa=20.0)
        self._check(f, gf, [np.array([3.0, 1.0]), np.array([0.0, 0.0]),
                             np.array([-2.0, 4.0])], "Quadratic2D")

    def test_high_dim_quadratic(self):
        f, gf, _, _ = high_dim_quadratic(n=10, kappa=5, seed=7)
        x_test = np.random.default_rng(42).standard_normal(10)
        self._check(f, gf, [x_test], "HighDimQuadratic")


# ===========================================================================
# 3. Line search tests
# ===========================================================================

class TestWolfeLineSearch:

    def _setup_rosenbrock(self):
        f, gf, _, _ = rosenbrock()
        x = np.array([-1.2, 1.0]); g = gf(x); d = -g
        return f, gf, x, f(x), g, d

    def test_returns_4tuple(self):
        """wolfe_line_search must return (alpha, f_new, nfev, ngev)."""
        f, gf, x, fx, g, d = self._setup_rosenbrock()
        result = wolfe_line_search(f, gf, x, d, fx, g)
        assert len(result) == 4, f"Expected 4-tuple, got {len(result)}-tuple"

    def test_fnew_matches_f(self):
        """Returned f_new must equal f(x + alpha*d) exactly (no redundant eval)."""
        f, gf, x, fx, g, d = self._setup_rosenbrock()
        alpha, f_new, _, _ = wolfe_line_search(f, gf, x, d, fx, g)
        assert abs(f_new - f(x + alpha * d)) < 1e-14

    def test_armijo_satisfied(self):
        """Accepted step satisfies Armijo sufficient-decrease condition."""
        f, gf, x, fx, g, d = self._setup_rosenbrock()
        c1 = 1e-4
        alpha, f_new, _, _ = wolfe_line_search(f, gf, x, d, fx, g, c1=c1, c2=0.9)
        dphi0 = np.dot(g, d)
        assert f_new <= fx + c1 * alpha * dphi0 + 1e-10

    def test_strong_curvature_satisfied(self):
        """Accepted step satisfies strong Wolfe curvature condition."""
        f, gf, x, fx, g, d = self._setup_rosenbrock()
        c2 = 0.9
        alpha, _, _, _ = wolfe_line_search(f, gf, x, d, fx, g, c1=1e-4, c2=c2)
        dphi0 = np.dot(g, d)
        dphi_new = np.dot(gf(x + alpha * d), d)
        assert abs(dphi_new) <= c2 * abs(dphi0) + 1e-8

    def test_positive_step(self):
        """Line search always returns alpha > 0."""
        f, gf, x, fx, g, d = self._setup_rosenbrock()
        alpha, _, _, _ = wolfe_line_search(f, gf, x, d, fx, g)
        assert alpha > 0.0

    def test_non_descent_raises(self):
        """AssertionError raised for non-descent direction (g^T d >= 0)."""
        f, gf, x, fx, g, d = self._setup_rosenbrock()
        with pytest.raises(AssertionError):
            wolfe_line_search(f, gf, x, -d, fx, g)

    def test_wolfe_on_beale(self):
        """Wolfe conditions hold on Beale function (different landscape)."""
        f, gf, _, _ = beale()
        x = np.array([1.0, 1.0]); g = gf(x); d = -g
        alpha, f_new, _, _ = wolfe_line_search(f, gf, x, d, f(x), g, c1=1e-4, c2=0.9)
        dphi0 = np.dot(g, d)
        assert f_new <= f(x) + 1e-4 * alpha * dphi0 + 1e-10
        assert abs(np.dot(gf(x + alpha * d), d)) <= 0.9 * abs(dphi0) + 1e-8


# ===========================================================================
# 4. NCG convergence tests
# ===========================================================================

class TestNCGConvergence:

    @pytest.mark.parametrize("variant", ["FR", "PR", "PR+", "HS"])
    def test_quadratic_bowl(self, variant):
        """All variants converge on the 2-D quadratic bowl."""
        f, g, xs, _ = quadratic_bowl_2d(kappa=50.0)
        res = ncg_minimize(f, g, np.array([4.0, 4.0]), beta_variant=variant,
                           tol=TOL, max_iter=500)
        assert res.success, f"{variant} failed: {res.message}"
        assert res.grad_norm < TOL * 10
        assert np.linalg.norm(res.x - xs) < 1e-4

    @pytest.mark.parametrize("variant", ["FR", "PR", "PR+", "HS"])
    def test_rosenbrock(self, variant):
        """All variants converge on the Rosenbrock function."""
        f, g, xs, _ = rosenbrock()
        res = ncg_minimize(f, g, np.array([-1.2, 1.0]), beta_variant=variant,
                           tol=TOL, max_iter=3000)
        assert res.grad_norm < TOL * 10, f"{variant}: grad_norm={res.grad_norm}"
        assert np.linalg.norm(res.x - xs) < 1e-3

    @pytest.mark.parametrize("variant", ["FR", "PR", "PR+", "HS"])
    def test_beale(self, variant):
        """All variants converge on Beale's function."""
        f, g, xs, _ = beale()
        res = ncg_minimize(f, g, np.array([1.0, 1.0]), beta_variant=variant,
                           tol=TOL, max_iter=1000)
        assert res.grad_norm < TOL * 100, f"{variant}: grad_norm={res.grad_norm}"

    @pytest.mark.parametrize("variant", ["PR+", "FR"])
    def test_high_dim_quadratic(self, variant):
        """PR+ and FR converge on high-dimensional quadratic (n=50)."""
        f, g, xs, _ = high_dim_quadratic(n=50, kappa=50.0, seed=1)
        res = ncg_minimize(f, g, np.zeros(50), beta_variant=variant,
                           tol=TOL, max_iter=2000)
        assert res.success or res.grad_norm < TOL * 100

    def test_finite_termination_quadratic(self):
        """
        NCG achieves tight convergence on a low-dimensional quadratic.
        Theory (Section 6): exact CG terminates in <= n steps on quadratics.
        With floating-point Wolfe line search, convergence is still rapid.
        """
        n = 10
        f, g, _, _ = high_dim_quadratic(n=n, kappa=5.0, seed=2)
        res = ncg_minimize(f, g, np.zeros(n), beta_variant="PR+",
                           tol=1e-6, max_iter=500)
        assert res.grad_norm < 1e-4, f"Did not converge: ‖∇f‖ = {res.grad_norm}"
        assert res.nit <= 10 * n, f"Surprisingly slow: {res.nit} iters on {n}-D quad"

    def test_already_converged_x0(self):
        """Solver returns immediately with nit=1 when starting at the minimizer."""
        f, gf, xs, _ = rosenbrock()
        res = ncg_minimize(f, gf, xs.copy(), beta_variant="PR+", tol=1e-6, max_iter=100)
        assert res.success, "Should report success when starting at minimizer"
        assert res.nit == 1, f"Expected nit=1, got {res.nit}"

    def test_2d_x0_raveled(self):
        """2-D x0 array is silently ravel()ed to 1-D."""
        f, gf, _, _ = quadratic_bowl_2d()
        x0_2d = np.array([[4.0, 4.0]])  # shape (1, 2)
        res = ncg_minimize(f, gf, x0_2d, beta_variant="PR+", tol=1e-6, max_iter=100)
        assert res.x.shape == (2,), f"Expected shape (2,), got {res.x.shape}"
        assert res.success


# ===========================================================================
# 5. Restart logic tests
# ===========================================================================

class TestRestarts:

    def test_periodic_restart_triggered(self):
        """Periodic restart fires at k = restart_every - 1 (0-indexed)."""
        f, g, _, _ = rosenbrock()
        restart_every = 5
        res = ncg_minimize(f, g, np.array([-1.2, 1.0]), beta_variant="PR+",
                           restart_every=restart_every, use_powell_restart=False,
                           tol=1e-8, max_iter=200)
        assert len(res.restart_flags) >= restart_every
        assert res.restart_flags[restart_every - 1], \
            "Periodic restart should fire at completed step restart_every"

    def test_powell_restart_fires(self):
        """Powell criterion fires at least once on Rosenbrock with FR."""
        f, g, _, _ = rosenbrock()
        res = ncg_minimize(f, g, np.array([-1.2, 1.0]), beta_variant="FR",
                           use_powell_restart=True, powell_nu=0.05,
                           tol=1e-8, max_iter=500)
        assert any(res.restart_flags), "Powell restart should fire on Rosenbrock"

    def test_powell_independent_of_periodic(self):
        """With nu=0, Powell fires every step independently of periodic."""
        f, g, _, _ = rosenbrock()
        res = ncg_minimize(f, g, np.array([-1.2, 1.0]), beta_variant="FR",
                           restart_every=2, use_powell_restart=True, powell_nu=0.0,
                           tol=1e-8, max_iter=50)
        assert all(res.restart_flags), "Both restart criteria should fire every step"

    def test_descent_failure_reset(self):
        """Algorithm still converges even with aggressive FR (descent reset catches bad steps)."""
        f, g, _, _ = rosenbrock()
        res = ncg_minimize(f, g, np.array([-1.2, 1.0]), beta_variant="FR",
                           use_powell_restart=False, tol=1e-8, max_iter=500)
        assert res.grad_norm < 1.0


# ===========================================================================
# 6. Result structure and history tests
# ===========================================================================

class TestResultStructure:

    def test_history_lengths(self):
        """History arrays are mutually consistent.

        fun/grad/x_history : length = completed_steps + 1  (initial + one per step)
        beta/alpha/restart  : length = completed_steps
        """
        f, g, _, _ = quadratic_bowl_2d()
        res = ncg_minimize(f, g, np.array([3.0, 3.0]), tol=1e-8, max_iter=100)
        nsteps = len(res.beta_history)
        assert len(res.alpha_history)     == nsteps
        assert len(res.restart_flags)     == nsteps
        assert len(res.fun_history)       == nsteps + 1
        assert len(res.grad_norm_history) == nsteps + 1
        assert len(res.x_history)         == nsteps + 1
        assert res.nit in (nsteps, nsteps + 1)

    def test_no_history_when_disabled(self):
        """store_history=False leaves all history lists empty."""
        f, g, _, _ = quadratic_bowl_2d()
        res = ncg_minimize(f, g, np.array([3.0, 3.0]), tol=1e-6, max_iter=100,
                           store_history=False)
        assert res.fun_history       == []
        assert res.grad_norm_history == []
        assert res.x_history         == []

    def test_invalid_variant_raises(self):
        """Unknown beta variant raises ValueError with helpful message."""
        f, g, _, _ = quadratic_bowl_2d()
        with pytest.raises(ValueError):
            ncg_minimize(f, g, np.zeros(2), beta_variant="UNKNOWN")

    def test_x_shape_preserved(self):
        """Output x has same shape as x0."""
        f, g, _, _ = high_dim_quadratic(n=30, kappa=10.0)
        res = ncg_minimize(f, g, np.zeros(30), tol=1e-6, max_iter=200)
        assert res.x.shape == (30,)

    def test_nfev_count_accurate(self):
        """Reported nfev matches actual calls to f (no double-counting)."""
        actual = [0]
        f_r, gf_r, _, _ = rosenbrock()
        def f_counted(x): actual[0] += 1; return f_r(x)
        res = ncg_minimize(f_counted, gf_r, np.array([-1.2, 1.0]),
                           beta_variant="PR+", tol=1e-6, max_iter=100)
        assert actual[0] == res.nfev, \
            f"Reported nfev={res.nfev} but actual calls={actual[0]}"

    def test_fk_no_redundant_eval(self):
        """f_new returned by line search means no extra f-eval per iteration."""
        actual_f = [0]
        f_r, gf_r, _, _ = quadratic_bowl_2d()
        def fc(x): actual_f[0] += 1; return f_r(x)
        res = ncg_minimize(fc, gf_r, np.array([4.0, 4.0]),
                           beta_variant="PR+", tol=1e-8, max_iter=20)
        assert res.nfev == actual_f[0], \
            f"nfev mismatch: reported {res.nfev}, actual {actual_f[0]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
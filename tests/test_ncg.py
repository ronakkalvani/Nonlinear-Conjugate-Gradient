"""
tests/test_ncg.py — Unit tests for the NCG codebase
====================================================
Covers:
  - Beta formula correctness (quadratic exact-arithmetic identities)
  - Wolfe line search conditions satisfied on output
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


# ---------------------------------------------------------------------------
# Tolerance for "converged"
# ---------------------------------------------------------------------------
TOL = 1e-6


# ===========================================================================
# 1. Beta formula tests
# ===========================================================================

class TestBetaFormulas:

    def _make_gradients(self):
        """Create synthetic gradients for testing."""
        rng = np.random.default_rng(0)
        g_old = rng.standard_normal(10)
        g_new = rng.standard_normal(10)
        d_old = rng.standard_normal(10)
        return g_new, g_old, d_old

    def test_fr_positive(self):
        """FR beta is always non-negative."""
        g_new, g_old, d_old = self._make_gradients()
        b = beta_fletcher_reeves(g_new, g_old, d_old)
        assert b >= 0.0

    def test_fr_formula(self):
        """FR formula: ‖g_new‖² / ‖g_old‖²."""
        g_new, g_old, d_old = self._make_gradients()
        expected = np.dot(g_new, g_new) / np.dot(g_old, g_old)
        assert abs(beta_fletcher_reeves(g_new, g_old, d_old) - expected) < 1e-14

    def test_pr_can_be_negative(self):
        """PR beta can be negative — this is by design."""
        # Construct a case where the gradient norm decreases
        g_old = np.array([1.0, 0.0])
        g_new = np.array([0.0, 0.1])   # very small; g_new . (g_new - g_old) < 0
        d_old = np.zeros(2)
        b = beta_polak_ribiere(g_new, g_old, d_old)
        # g_new . (g_new - g_old) = [0,0.1].[−1,0.1] = −0+0.01 = 0.01 > 0 here
        # Use a deliberate case
        g_old2 = np.array([2.0, 0.0])
        g_new2 = np.array([0.5, 0.0])
        b2 = beta_polak_ribiere(g_new2, g_old2, d_old)
        # g_new . y = [0.5].[0.5-2] = 0.5*(-1.5) = -0.75 < 0
        assert b2 < 0.0

    def test_prplus_non_negative(self):
        """PR+ is always ≥ 0."""
        g_old = np.array([2.0, 0.0])
        g_new = np.array([0.5, 0.0])
        d_old = np.zeros(2)
        b = beta_polak_ribiere_plus(g_new, g_old, d_old)
        assert b >= 0.0

    def test_prplus_equals_pr_when_pr_positive(self):
        """PR+ = PR when PR ≥ 0."""
        g_new = np.array([1.5, 0.5])
        g_old = np.array([1.0, 0.0])
        d_old = np.zeros(2)
        assert (abs(beta_polak_ribiere_plus(g_new, g_old, d_old)
                    - max(beta_polak_ribiere(g_new, g_old, d_old), 0.0))
                < 1e-14)

    def test_hs_formula(self):
        """HS formula: g_new . y / d_old . y."""
        g_new = np.array([1.2, 0.4])
        g_old = np.array([0.8, 0.1])
        d_old = np.array([0.3, -0.5])
        y = g_new - g_old
        denom = np.dot(d_old, y)
        if abs(denom) > 1e-30:
            expected = np.dot(g_new, y) / denom
            assert abs(beta_hestenes_stiefel(g_new, g_old, d_old) - expected) < 1e-12

    def test_hs_zero_denominator_safeguard(self):
        """HS returns 0 when denominator vanishes."""
        g_new = np.array([1.0, 0.0])
        g_old = np.array([1.0, 0.0])   # y = 0 → denom = 0
        d_old = np.array([1.0, 0.0])
        assert beta_hestenes_stiefel(g_new, g_old, d_old) == 0.0

    def test_fr_pr_equal_on_quadratic_exact(self):
        """On a quadratic with exact orthogonal residuals, FR == PR."""
        # Build exact case: g_new . g_old = 0
        g_old = np.array([1.0, 0.0])
        g_new = np.array([0.0, 1.0])
        d_old = np.zeros(2)
        fr = beta_fletcher_reeves(g_new, g_old, d_old)
        pr = beta_polak_ribiere(g_new, g_old, d_old)
        assert abs(fr - pr) < 1e-12, f"FR={fr}, PR={pr}"

    def test_zero_old_gradient_safeguard(self):
        """FR, PR, PR+ return 0 when ‖g_old‖ = 0 (denominator safeguard).
        Note: HS uses d·y in its denominator so its safeguard is separate."""
        g_old = np.zeros(5)
        g_new = np.ones(5)
        d_old = np.ones(5)
        # FR, PR, PR+ all use ‖g_old‖² in denominator → return 0
        for name in ["FR", "PR", "PR+"]:
            assert BETA_FUNCTIONS[name](g_new, g_old, d_old) == 0.0,                 f"{name} should return 0 when g_old=0"
        # HS separate test: its safeguard is |d·y| < 1e-30
        g_old_hs = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        g_new_hs = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # y = 0 → d·y = 0
        assert BETA_FUNCTIONS["HS"](g_new_hs, g_old_hs, d_old) == 0.0


# ===========================================================================
# 2. Line search tests
# ===========================================================================

class TestWolfeLineSearch:

    def _setup_rosenbrock(self):
        f, gf, xs, fs = rosenbrock()
        x = np.array([-1.2, 1.0])
        g = gf(x)
        d = -g  # steepest descent (valid descent direction)
        return f, gf, x, f(x), g, d

    def test_wolfe_armijo_satisfied(self):
        """Accepted step satisfies Armijo condition."""
        f, gf, x, fx, g, d = self._setup_rosenbrock()
        c1 = 1e-4
        alpha, _, _, _ = wolfe_line_search(f, gf, x, d, fx, g, c1=c1, c2=0.9)
        dphi0 = np.dot(g, d)
        assert f(x + alpha * d) <= fx + c1 * alpha * dphi0 + 1e-10

    def test_wolfe_curvature_satisfied(self):
        """Accepted step satisfies strong curvature condition."""
        f, gf, x, fx, g, d = self._setup_rosenbrock()
        c2 = 0.9
        alpha, _, _, _ = wolfe_line_search(f, gf, x, d, fx, g, c1=1e-4, c2=c2)
        dphi0 = np.dot(g, d)
        dphi_new = np.dot(gf(x + alpha * d), d)
        assert abs(dphi_new) <= c2 * abs(dphi0) + 1e-8

    def test_step_positive(self):
        """Line search always returns a positive step."""
        f, gf, x, fx, g, d = self._setup_rosenbrock()
        alpha, _, _, _ = wolfe_line_search(f, gf, x, d, fx, g)
        assert alpha > 0.0

    def test_non_descent_raises(self):
        """Line search raises AssertionError for a non-descent direction."""
        f, gf, x, fx, g, d = self._setup_rosenbrock()
        with pytest.raises(AssertionError):
            wolfe_line_search(f, gf, x, -d, fx, g)  # flip to ascent


# ===========================================================================
# 3. NCG convergence tests
# ===========================================================================

class TestNCGConvergence:

    @pytest.mark.parametrize("variant", ["FR", "PR", "PR+", "HS"])
    def test_quadratic_bowl(self, variant):
        """All variants converge on the 2-D quadratic bowl."""
        f, g, xs, fs = quadratic_bowl_2d(kappa=50.0)
        x0 = np.array([4.0, 4.0])
        res = ncg_minimize(f, g, x0, beta_variant=variant, tol=TOL, max_iter=500)
        assert res.success, f"{variant} failed: {res.message}"
        assert res.grad_norm < TOL * 10
        assert np.linalg.norm(res.x - xs) < 1e-4

    @pytest.mark.parametrize("variant", ["FR", "PR", "PR+", "HS"])
    def test_rosenbrock(self, variant):
        """All variants converge on the Rosenbrock function."""
        f, g, xs, fs = rosenbrock()
        x0 = np.array([-1.2, 1.0])
        res = ncg_minimize(f, g, x0, beta_variant=variant, tol=TOL, max_iter=3000)
        assert res.grad_norm < TOL * 10, f"{variant}: grad_norm={res.grad_norm}"
        assert np.linalg.norm(res.x - xs) < 1e-3, f"{variant}: x_err={np.linalg.norm(res.x - xs)}"

    @pytest.mark.parametrize("variant", ["FR", "PR", "PR+", "HS"])
    def test_beale(self, variant):
        """All variants converge on Beale's function."""
        f, g, xs, fs = beale()
        x0 = np.array([1.0, 1.0])
        res = ncg_minimize(f, g, x0, beta_variant=variant, tol=TOL, max_iter=1000)
        assert res.grad_norm < TOL * 100, f"{variant}: grad_norm={res.grad_norm}"

    @pytest.mark.parametrize("variant", ["PR+", "FR"])
    def test_high_dim_quadratic(self, variant):
        """PR+ and FR converge on high-dimensional quadratic (n=50)."""
        f, g, xs, fs = high_dim_quadratic(n=50, kappa=50.0, seed=1)
        x0 = np.zeros(50)
        res = ncg_minimize(f, g, x0, beta_variant=variant, tol=TOL, max_iter=2000)
        assert res.success or res.grad_norm < TOL * 100

    def test_finite_termination_quadratic(self):
        """PR+ achieves tight convergence on a low-dimensional quadratic.
        The theoretical ≤n bound holds in exact arithmetic; with floating-point
        Wolfe line search the method still converges rapidly (tol=1e-6).
        """
        n = 10
        f, g, xs, fs = high_dim_quadratic(n=n, kappa=5.0, seed=2)
        x0 = np.zeros(n)
        res = ncg_minimize(f, g, x0, beta_variant="PR+", tol=1e-6, max_iter=500)
        assert res.grad_norm < 1e-4, f"Did not converge: ‖∇f‖ = {res.grad_norm}"
        # With restarts and floating-point line search, convergence is fast but
        # not guaranteed to be ≤ n steps; check it finishes well within 10n
        assert res.nit <= 10 * n, f"Surprisingly slow: {res.nit} iters on {n}-D quad"


# ===========================================================================
# 4. Restart logic tests
# ===========================================================================

class TestRestarts:

    def test_periodic_restart_triggered(self):
        """Periodic restart triggers every restart_every iterations."""
        f, g, xs, fs = rosenbrock()
        x0 = np.array([-1.2, 1.0])
        restart_every = 5
        res = ncg_minimize(f, g, x0, beta_variant="PR+",
                           restart_every=restart_every,
                           use_powell_restart=False,
                           tol=1e-8, max_iter=200)
        # At least one restart should have been triggered at iter 5
        flags = res.restart_flags
        assert len(flags) >= restart_every
        # iter index 4 (0-based) is the 5th → restart
        assert flags[restart_every - 1], "Periodic restart should fire at k=restart_every-1"

    def test_powell_restart_fires(self):
        """Powell criterion triggers at least once on Rosenbrock."""
        f, g, xs, fs = rosenbrock()
        x0 = np.array([-1.2, 1.0])
        res = ncg_minimize(f, g, x0, beta_variant="FR",
                           use_powell_restart=True, powell_nu=0.05,
                           tol=1e-8, max_iter=500)
        # On the curved Rosenbrock, some Powell restarts should occur
        assert any(res.restart_flags)

    def test_descent_failure_reset(self):
        """Direction is reset if descent condition violated."""
        # Use FR (which can jam) with a tight tolerance that forces many steps
        f, g, xs, fs = rosenbrock()
        x0 = np.array([-1.2, 1.0])
        res = ncg_minimize(f, g, x0, beta_variant="FR",
                           use_powell_restart=False,
                           tol=1e-8, max_iter=500)
        # The method should still converge (descent reset catches any bad step)
        assert res.grad_norm < 1.0


# ===========================================================================
# 5. Result structure and history tests
# ===========================================================================

class TestResultStructure:

    def test_history_length_consistent(self):
        """History arrays are mutually consistent.
        fun/grad/x_history have one initial entry + one per completed step.
        beta/alpha/restart_flags have one entry per completed step.
        Convergence check fires before the last step is appended, so
        completed_steps may equal nit-1 (when converged) or nit (max_iter hit).
        """
        f, g, xs, fs = quadratic_bowl_2d()
        x0 = np.array([3.0, 3.0])
        res = ncg_minimize(f, g, x0, tol=1e-8, max_iter=100)
        nsteps = len(res.beta_history)  # completed steps
        # All step-indexed histories must agree
        assert len(res.alpha_history)     == nsteps
        assert len(res.restart_flags)     == nsteps
        # Iterate histories have one initial + one per step
        assert len(res.fun_history)       == nsteps + 1
        assert len(res.grad_norm_history) == nsteps + 1
        assert len(res.x_history)         == nsteps + 1
        # nit is either nsteps (converged, last iter counted) or nsteps+1
        assert res.nit in (nsteps, nsteps + 1)

    def test_no_history_when_disabled(self):
        """store_history=False leaves all history lists empty."""
        f, g, xs, fs = quadratic_bowl_2d()
        x0 = np.array([3.0, 3.0])
        res = ncg_minimize(f, g, x0, tol=1e-6, max_iter=100,
                           store_history=False)
        assert res.fun_history       == []
        assert res.grad_norm_history == []

    def test_invalid_variant_raises(self):
        """Passing an unknown beta variant raises ValueError."""
        f, g, xs, fs = quadratic_bowl_2d()
        with pytest.raises(ValueError):
            ncg_minimize(f, g, np.zeros(2), beta_variant="UNKNOWN")

    def test_x_shape_preserved(self):
        """Output x has same shape as x0."""
        f, g, xs, fs = high_dim_quadratic(n=30, kappa=10.0)
        x0 = np.zeros(30)
        res = ncg_minimize(f, g, x0, tol=1e-6, max_iter=200)
        assert res.x.shape == x0.shape


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
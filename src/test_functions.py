"""
test_functions.py — Benchmark functions for NCG testing
========================================================
Implements all four test functions specified in the checklist (Section 7):
  1. Quadratic bowl (2-D)          — verifies finite-step property
  2. Rosenbrock function           — classic nonlinear test
  3. Beale's function              — multiple local structures
  4. High-dimensional quadratic    — tests finite-step termination

Each function returns (f, grad_f, x_star, f_star) so that convergence
can be measured against the known minimizer and minimum value.

All gradients are verified against numerical central differences in run_tests.py.
"""

import numpy as np
from typing import Tuple, Callable

FuncType = Callable[[np.ndarray], float]
GradType = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# 1. Quadratic bowl  f(x) = ½ x^T A x - b^T x
# ---------------------------------------------------------------------------

def make_quadratic(A: np.ndarray, b: np.ndarray
                   ) -> Tuple[FuncType, GradType, np.ndarray, float]:
    """
    Factory for a strictly-convex quadratic.

        f(x)   = ½ x^T A x - b^T x
        ∇f(x)  = Ax - b
        x*     = A^{-1} b
        f(x*)  = -½ b^T A^{-1} b

    Parameters
    ----------
    A : (n×n) symmetric positive-definite matrix
    b : (n,) right-hand side vector
    """
    A = np.array(A, dtype=float); b = np.array(b, dtype=float)
    x_star = np.linalg.solve(A, b)
    f_star = float(-0.5 * b @ x_star)

    def f(x):
        x = np.asarray(x, dtype=float); return 0.5 * x @ A @ x - b @ x

    def grad_f(x):
        x = np.asarray(x, dtype=float); return A @ x - b

    return f, grad_f, x_star, f_star


def quadratic_bowl_2d(kappa: float = 50.0
                      ) -> Tuple[FuncType, GradType, np.ndarray, float]:
    """
    Ill-conditioned 2-D quadratic bowl.

        f(x, y) = ½ x² + (κ/2) y²

    Condition number κ (default 50) exposes the zig-zagging of gradient
    descent vs. the efficient path of NCG (see Figure 1 in report).

    Minimizer: x* = (0, 0),  f* = 0.
    """
    A = np.diag([1.0, kappa]); b = np.zeros(2)
    return make_quadratic(A, b)


# ---------------------------------------------------------------------------
# 2. Rosenbrock  f(x, y) = (1 - x)² + 100(y - x²)²
# ---------------------------------------------------------------------------

def rosenbrock() -> Tuple[FuncType, GradType, np.ndarray, float]:
    """
    Classic Rosenbrock banana function.

        f(x, y) = (1 - x)² + 100(y - x²)²

    The narrow curved valley makes this a canonical hard test for NCG.
    Starting point: x0 = (-1.2, 1.0).
    Minimizer: x* = (1, 1),  f* = 0.
    """
    def f(x):
        x = np.asarray(x, dtype=float)
        return (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2

    def grad_f(x):
        x = np.asarray(x, dtype=float)
        dfdx = -2.0*(1.0 - x[0]) - 400.0*x[0]*(x[1] - x[0]**2)
        dfdy =  200.0*(x[1] - x[0]**2)
        return np.array([dfdx, dfdy])

    return f, grad_f, np.array([1.0, 1.0]), 0.0


# ---------------------------------------------------------------------------
# 3. Beale's function
# ---------------------------------------------------------------------------

def beale() -> Tuple[FuncType, GradType, np.ndarray, float]:
    """
    Beale's function.

        f(x, y) = (1.5 - x + xy)²
                + (2.25 - x + xy²)²
                + (2.625 - x + xy³)²

    Defined on [-4.5, 4.5]².
    Minimizer: x* = (3, 0.5),  f* = 0.
    """
    def f(x):
        x = np.asarray(x, dtype=float)
        t1 = 1.5   - x[0] + x[0]*x[1]
        t2 = 2.25  - x[0] + x[0]*x[1]**2
        t3 = 2.625 - x[0] + x[0]*x[1]**3
        return t1**2 + t2**2 + t3**2

    def grad_f(x):
        x = np.asarray(x, dtype=float)
        t1 = 1.5   - x[0] + x[0]*x[1]
        t2 = 2.25  - x[0] + x[0]*x[1]**2
        t3 = 2.625 - x[0] + x[0]*x[1]**3
        # ∂f/∂x_0
        dt1_dx0 = -1.0 + x[1]; dt2_dx0 = -1.0 + x[1]**2; dt3_dx0 = -1.0 + x[1]**3
        dfdx0 = 2.0*(t1*dt1_dx0 + t2*dt2_dx0 + t3*dt3_dx0)
        # ∂f/∂x_1
        dt1_dx1 = x[0]; dt2_dx1 = 2.0*x[0]*x[1]; dt3_dx1 = 3.0*x[0]*x[1]**2
        dfdx1 = 2.0*(t1*dt1_dx1 + t2*dt2_dx1 + t3*dt3_dx1)
        return np.array([dfdx0, dfdx1])

    return f, grad_f, np.array([3.0, 0.5]), 0.0


# ---------------------------------------------------------------------------
# 4. High-dimensional quadratic (n-dimensional, random SPD matrix)
# ---------------------------------------------------------------------------

def high_dim_quadratic(n: int = 100, kappa: float = 100.0,
                        seed: int = 42
                        ) -> Tuple[FuncType, GradType, np.ndarray, float]:
    """
    High-dimensional quadratic with prescribed condition number.

        f(x) = ½ x^T A x - b^T x

    A is constructed with eigenvalues log-spaced between 1 and κ so that
    cond(A) = κ. This is the canonical setting for proving NCG terminates
    in n steps in exact arithmetic (Section 6 of the report).

    Parameters
    ----------
    n     : problem dimension
    kappa : condition number of A (default 100)
    seed  : random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    eigvals = np.logspace(0, np.log10(kappa), n)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = Q @ np.diag(eigvals) @ Q.T
    b = rng.standard_normal(n)
    return make_quadratic(A, b)


# ---------------------------------------------------------------------------
# Convenience: list all built-in test problems
# ---------------------------------------------------------------------------

def get_all_test_problems():
    """
    Returns a list of dicts describing each test problem.
    Keys: name, f, grad_f, x0, x_star, f_star, dim
    """
    problems = []

    f, g, xs, fs = quadratic_bowl_2d(kappa=50.0)
    problems.append(dict(name="Quadratic Bowl 2D (κ=50)", f=f, grad_f=g,
                         x0=np.array([4.0, 4.0]), x_star=xs, f_star=fs, dim=2))

    f, g, xs, fs = rosenbrock()
    problems.append(dict(name="Rosenbrock", f=f, grad_f=g,
                         x0=np.array([-1.2, 1.0]), x_star=xs, f_star=fs, dim=2))

    f, g, xs, fs = beale()
    problems.append(dict(name="Beale", f=f, grad_f=g,
                         x0=np.array([1.0, 1.0]), x_star=xs, f_star=fs, dim=2))

    f, g, xs, fs = high_dim_quadratic(n=100, kappa=100.0)
    problems.append(dict(name="High-Dim Quadratic (n=100, κ=100)", f=f, grad_f=g,
                         x0=np.zeros(100), x_star=xs, f_star=fs, dim=100))

    return problems
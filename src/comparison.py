"""
comparison.py — Benchmark NCG variants against Gradient Descent and scipy
=========================================================================
Runs all four NCG beta variants, a pure Gradient Descent baseline, and
scipy.optimize.minimize (L-BFGS-B) on every test problem, returning a
unified comparison table.

Alignments with report (Sections 5, 6, 10, 11):
  - GD success threshold: grad_norm < tol (strict, matching NCG convergence def)
  - scipy: uses gtol=tol (gradient norm stopping); ftol=0 to disable function
    tolerance so only gradient norm controls termination (Nocedal & Wright §7.2)
  - GD x_history: appended AFTER update (index 0 = initial, index k = after step k)
    matching NCG's convention for consistent plot comparisons
"""

import numpy as np
import time
from typing import List
from dataclasses import dataclass

import scipy.optimize as opt

from ncg import ncg_minimize, NCGResult
from test_functions import get_all_test_problems


# ---------------------------------------------------------------------------
# Gradient Descent baseline (backtracking Armijo line search)
# ---------------------------------------------------------------------------

def gradient_descent(f, grad_f, x0, tol=1e-6, max_iter=10000,
                     alpha_init=1.0, rho=0.5, c1=1e-4):
    """
    Steepest-descent with backtracking Armijo line search.

    Used as a performance baseline to highlight NCG's advantages on
    ill-conditioned problems (see Section 11, Comparison with GD).

    History convention: x_history[0] = x0 (before any step),
    x_history[k+1] = iterate after step k. This matches NCG convention
    so histories can be compared directly in plots.
    """
    x = np.array(x0, dtype=float).copy()
    nfev = ngev = 0
    fun_hist = []; gnorm_hist = []; x_hist = []

    fk = f(x);       nfev += 1
    gk = grad_f(x);  ngev += 1
    fun_hist.append(fk); gnorm_hist.append(float(np.linalg.norm(gk))); x_hist.append(x.copy())

    for k in range(max_iter):
        gnorm = float(np.linalg.norm(gk))
        if gnorm < tol:
            break

        d = -gk
        alpha = alpha_init
        dphi0 = np.dot(gk, d)  # = -||gk||^2 < 0
        f_trial = fk

        for _ in range(50):
            nfev += 1
            f_trial = f(x + alpha * d)
            if f_trial <= fk + c1 * alpha * dphi0:
                break
            alpha *= rho

        # Append AFTER update — consistent with NCG convention
        x   = x + alpha * d
        fk  = f_trial
        gk  = grad_f(x); ngev += 1
        fun_hist.append(fk)
        gnorm_hist.append(float(np.linalg.norm(gk)))
        x_hist.append(x.copy())

    return dict(x=x, fun=fk, grad_norm=float(np.linalg.norm(gk)),
                nit=k + 1, nfev=nfev, ngev=ngev,
                fun_history=fun_hist, grad_norm_history=gnorm_hist,
                x_history=x_hist, method="GD")


# ---------------------------------------------------------------------------
# scipy wrapper
# ---------------------------------------------------------------------------

def run_scipy(f, grad_f, x0, tol=1e-6, method="L-BFGS-B"):
    """
    Thin wrapper around scipy.optimize.minimize for comparison.

    Uses gtol=tol (gradient norm) as the primary stopping criterion.
    ftol=0 disables the function-value tolerance so only the gradient
    norm controls termination — consistent with the convergence definition
    ||∇f(xk)|| → 0 used throughout the report (Section 5).
    """
    nfev = [0]; ngev = [0]

    def f_counted(x): nfev[0] += 1; return f(x)
    def g_counted(x): ngev[0] += 1; return grad_f(x)

    t0 = time.perf_counter()
    res = opt.minimize(f_counted, x0.copy(), jac=g_counted, method=method,
                       options={"gtol": tol, "ftol": 0, "maxiter": 10000})
    elapsed = time.perf_counter() - t0

    return dict(x=res.x, fun=res.fun,
                grad_norm=float(np.linalg.norm(grad_f(res.x))),
                nit=res.nit, nfev=nfev[0], ngev=ngev[0],
                success=res.success, message=res.message,
                elapsed=elapsed, method=f"scipy({method})")


# ---------------------------------------------------------------------------
# Main comparison runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRow:
    problem:   str
    method:    str
    nit:       int
    nfev:      int
    ngev:      int
    fun:       float
    grad_norm: float
    success:   bool
    elapsed_s: float
    x_error:   float   # ||x - x*||


def run_all_benchmarks(tol: float = 1e-6) -> List[BenchmarkRow]:
    """Run every method on every problem and collect BenchmarkRow results."""
    problems = get_all_test_problems()
    rows: List[BenchmarkRow] = []

    for prob in problems:
        name = prob["name"]; f = prob["f"]; gf = prob["grad_f"]
        x0 = prob["x0"];     x_star = prob["x_star"]

        print(f"\n{'='*60}\n  {name}\n{'='*60}")

        for variant in ["FR", "PR", "PR+", "HS"]:
            t0  = time.perf_counter()
            res = ncg_minimize(f, gf, x0.copy(), beta_variant=variant,
                               tol=tol, max_iter=5000)
            elapsed = time.perf_counter() - t0
            xerr = float(np.linalg.norm(res.x - x_star))
            print(f"  NCG-{variant:3s}  iter={res.nit:5d}  f={res.fun:+.3e}  "
                  f"||grad_f||={res.grad_norm:.2e}  ||x-x*||={xerr:.2e}  "
                  f"{'OK' if res.success else '--'}")
            rows.append(BenchmarkRow(problem=name, method=f"NCG-{variant}",
                nit=res.nit, nfev=res.nfev, ngev=res.ngev,
                fun=res.fun, grad_norm=res.grad_norm,
                success=res.success, elapsed_s=elapsed, x_error=xerr))

        t0  = time.perf_counter()
        gd  = gradient_descent(f, gf, x0.copy(), tol=tol, max_iter=5000)
        elapsed = time.perf_counter() - t0
        xerr = float(np.linalg.norm(gd["x"] - x_star))
        gd_success = gd["grad_norm"] < tol  # strict: same criterion as NCG
        print(f"  GD       iter={gd['nit']:5d}  f={gd['fun']:+.3e}  "
              f"||grad_f||={gd['grad_norm']:.2e}  ||x-x*||={xerr:.2e}  "
              f"{'OK' if gd_success else '--'}")
        rows.append(BenchmarkRow(problem=name, method="GradDesc",
            nit=gd["nit"], nfev=gd["nfev"], ngev=gd["ngev"],
            fun=gd["fun"], grad_norm=gd["grad_norm"],
            success=gd_success, elapsed_s=elapsed, x_error=xerr))

        sp   = run_scipy(f, gf, x0.copy(), tol=tol)
        xerr = float(np.linalg.norm(sp["x"] - x_star))
        print(f"  scipy    iter={sp['nit']:5d}  f={sp['fun']:+.3e}  "
              f"||grad_f||={sp['grad_norm']:.2e}  ||x-x*||={xerr:.2e}  "
              f"{'OK' if sp['success'] else '--'}")
        rows.append(BenchmarkRow(problem=name, method="scipy(L-BFGS-B)",
            nit=sp["nit"], nfev=sp["nfev"], ngev=sp["ngev"],
            fun=sp["fun"], grad_norm=sp["grad_norm"],
            success=sp["success"], elapsed_s=sp["elapsed"], x_error=xerr))

    return rows


def print_summary_table(rows: List[BenchmarkRow]) -> None:
    """Pretty-print benchmark results."""
    print("\n\n" + "="*92 + "\nSUMMARY TABLE\n" + "="*92)
    header = (f"{'Problem':<40} {'Method':<18} {'Iter':>6} "
              f"{'NFev':>6} {'||grad_f||':>12} {'||x-x*||':>10} {'OK':>4}")
    print(header); print("-"*92)
    prev_prob = ""
    for r in rows:
        sep = "" if r.problem == prev_prob else "\n"
        print(sep + f"{r.problem:<40} {r.method:<18} {r.nit:>6} "
              f"{r.nfev:>6} {r.grad_norm:>12.2e} "
              f"{r.x_error:>10.2e} {'OK' if r.success else '--':>4}")
        prev_prob = r.problem
    print("="*92)
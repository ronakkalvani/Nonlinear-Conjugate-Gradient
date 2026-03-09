"""
plots.py — Convergence Demonstration Plots for NCG
===================================================
Generates all figures described in the To-Do Checklist §7.

Scaling design decisions (motivated by benchmark results):
  - All iteration-axis plots use LOG x-scale so that methods converging in
    9 iterations (NCG on quadratic) and 5000 iterations (GD on Rosenbrock)
    are both legible on the same axes.
  - Comparison bar chart uses LOG y-scale for the same reason.
  - Finite-termination study uses log-log axes (both κ and iteration count
    span orders of magnitude).
  - Methods that hit max_iter without converging are shown with a dashed
    marker at the end so the plot is not dominated by their flat tail.
  - Dashboard panels share the same log x-axis convention for consistency.
  - Trajectory plots are unaffected (spatial, not iteration-indexed).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import os, sys, warnings

sys.path.insert(0, os.path.dirname(__file__))
from ncg import ncg_minimize, NCGResult
from test_functions import (quadratic_bowl_2d, rosenbrock, beale,
                             high_dim_quadratic, get_all_test_problems)
from comparison import gradient_descent, run_scipy

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

VARIANTS    = ["FR", "PR", "PR+", "HS"]
COLORS      = {"FR": "#e63946", "PR": "#457b9d", "PR+": "#2a9d8f",
               "HS": "#e9c46a", "GD": "#6d6875", "scipy": "#264653"}
LINESTYLES  = {"FR": "-",  "PR": "--", "PR+": "-.", "HS": ":",
               "GD": (0,(3,1,1,1)), "scipy": "-"}
LINEWIDTHS  = {"FR": 2.0, "PR": 2.0, "PR+": 2.2, "HS": 2.0,
               "GD": 1.6, "scipy": 1.8}

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi":     130,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _history(result, key):
    """Extract a history list from either NCGResult or GD/scipy dict."""
    if isinstance(result, NCGResult):
        return getattr(result, key, [])
    return result.get(key, [])


def _iters(h):
    """0-based x-axis for a history of length len(h)."""
    return np.arange(len(h))


def _converged(result):
    if isinstance(result, NCGResult):
        return result.success
    return result.get("grad_norm", 1.0) < 1e-5


def _nit(result):
    if isinstance(result, NCGResult):
        return result.nit
    return result.get("nit", 0)


def _style(m):
    return dict(color=COLORS[m], linestyle=LINESTYLES[m],
                linewidth=LINEWIDTHS[m], alpha=0.88)


def _add_log_xaxis(ax, label="Iteration"):
    """Apply log x-axis with integer ticks that look clean."""
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else f"{x:.1f}"))
    ax.set_xlabel(label)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.grid(True, which="major", ls=":", alpha=0.6)


def _mark_unconverged(ax, result, x_val, y_val, color):
    """Put a small '×' at the end of a line that hit max_iter."""
    if not _converged(result) and y_val is not None and np.isfinite(y_val):
        ax.plot(x_val, y_val, "x", color=color, markersize=9,
                markeredgewidth=2, zorder=8, alpha=0.8)


def _savefig(fig, fname):
    fig.savefig(fname, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Run all methods on one problem
# ---------------------------------------------------------------------------

def _run_all(prob, tol=1e-8):
    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for v in VARIANTS:
            results[v] = ncg_minimize(prob["f"], prob["grad_f"],
                                      prob["x0"].copy(),
                                      beta_variant=v, tol=tol, max_iter=5000)
        results["GD"]    = gradient_descent(prob["f"], prob["grad_f"],
                                            prob["x0"].copy(), tol=tol, max_iter=5000)
        results["scipy"] = run_scipy(prob["f"], prob["grad_f"],
                                     prob["x0"].copy(), tol=tol)
    return results


# ---------------------------------------------------------------------------
# Plot 1: Loss vs iterations — LOG x, LOG y
# ---------------------------------------------------------------------------

def plot_loss_vs_iterations(prob, results, tag=""):
    """
    f(xk) - f* vs iterations on doubly-log axes.

    Log x: handles the 9-vs-5000 iteration spread cleanly.
    Log y: shows the exponential convergence rate characteristic of NCG.
    Two panels: NCG variants only (left) vs all methods (right).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Loss vs. Iterations — {prob['name']}", fontweight="bold", y=1.01)

    for ax, methods, title in zip(
        axes,
        [VARIANTS, ["GD", "scipy"] + VARIANTS],
        ["NCG Variants", "All Methods"]
    ):
        for m in methods:
            r = results[m]
            fh = _history(r, "fun_history")
            if not fh:
                continue
            f_min = prob["f_star"]
            shifted = np.maximum(np.array(fh, dtype=float) - f_min, 1e-20)
            xs = _iters(shifted)
            # Start from 1 (log x requires x > 0)
            ax.semilogy(xs + 1, shifted, **_style(m),
                        label=f"{m} ({_nit(r)} it)" if _converged(r)
                              else f"{m} (≥{_nit(r)})")
            _mark_unconverged(ax, r, xs[-1] + 1, shifted[-1], COLORS[m])

        ax.set_title(title)
        ax.set_ylabel("f(xₖ) − f*")
        _add_log_xaxis(ax)
        ax.legend(loc="upper right", framealpha=0.85)

    plt.tight_layout()
    _savefig(fig, os.path.join(RESULTS_DIR, f"loss_vs_iter_{tag}.png"))


# ---------------------------------------------------------------------------
# Plot 2: Gradient norm vs iterations — LOG x, LOG y
# ---------------------------------------------------------------------------

def plot_gradient_norm(prob, results, tag="", run_tol=1e-8):
    """
    ||∇f(xk)|| vs iterations on doubly-log axes.

    The convergence tolerance line helps visually confirm which methods
    actually crossed the threshold within the iteration budget.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Gradient Norm vs. Iterations — {prob['name']}",
                 fontweight="bold", y=1.01)

    for ax, methods, title in zip(
        axes,
        [VARIANTS, ["GD", "scipy"] + VARIANTS],
        ["NCG Variants", "All Methods"]
    ):
        for m in methods:
            r = results[m]
            gh = _history(r, "grad_norm_history")
            if not gh:
                continue
            xs = _iters(gh)
            ax.semilogy(xs + 1, gh, **_style(m),
                        label=f"{m} ({_nit(r)} it)" if _converged(r)
                              else f"{m} (≥{_nit(r)})")
            _mark_unconverged(ax, r, xs[-1] + 1, gh[-1], COLORS[m])

        ax.axhline(run_tol, color="black", ls="--", lw=1.2,
                   label=f"tol = {run_tol:.0e}")
        ax.set_title(title)
        ax.set_ylabel("‖∇f(xₖ)‖")
        _add_log_xaxis(ax)
        ax.legend(loc="upper right", framealpha=0.85)

    plt.tight_layout()
    _savefig(fig, os.path.join(RESULTS_DIR, f"grad_norm_{tag}.png"))


# ---------------------------------------------------------------------------
# Plot 3: 2-D trajectory on contour plot
# ---------------------------------------------------------------------------

def plot_2d_trajectory(prob, results, tag="", xlim=None, ylim=None):
    """Spatial path of iterates overlaid on contour plot (2-D problems only)."""
    if prob["dim"] != 2:
        return

    f = prob["f"]

    all_xh = []
    for r in results.values():
        xh = _history(r, "x_history")
        if xh:
            all_xh.extend(xh)

    if not all_xh:
        return

    pts = np.array(all_xh)
    x0_start = prob["x0"]
    pad = 0.5

    if xlim is None:
        xlim = (min(pts[:,0].min(), x0_start[0]) - pad,
                max(pts[:,0].max(), x0_start[0]) + pad)
    if ylim is None:
        ylim = (min(pts[:,1].min(), x0_start[1]) - pad,
                max(pts[:,1].max(), x0_start[1]) + pad)

    gx = np.linspace(xlim[0], xlim[1], 300)
    gy = np.linspace(ylim[0], ylim[1], 300)
    X, Y = np.meshgrid(gx, gy)
    Z = np.vectorize(lambda x, y: f(np.array([x, y])))(X, Y)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"2-D Trajectory — {prob['name']}", fontweight="bold", y=1.01)

    for ax, method_group, title in zip(
        axes,
        [["FR", "PR", "PR+", "HS"], ["GD", "PR+"]],
        ["NCG Variants", "NCG-PR+ vs Gradient Descent"]
    ):
        cs = ax.contourf(X, Y, np.log1p(Z - Z.min()), levels=40,
                         cmap="viridis", alpha=0.65)
        ax.contour(X, Y, np.log1p(Z - Z.min()), levels=20,
                   colors="white", linewidths=0.35, alpha=0.35)
        plt.colorbar(cs, ax=ax, shrink=0.82, label="log(1 + f − fmin)")

        for m in method_group:
            r = results[m]
            xh = _history(r, "x_history")
            if not xh:
                continue
            path = np.array(xh)
            n_pts = len(path)
            label = f"{m} ({n_pts-1} steps)"
            sty_traj = {k: v for k, v in _style(m).items()
                        if k not in ("linewidth", "alpha")}
            ax.plot(path[:,0], path[:,1], "o-", markersize=2.5,
                    linewidth=1.6, label=label, alpha=0.9, zorder=5,
                    **sty_traj)

        ax.scatter(*x0_start, s=130, c="white", edgecolors="black",
                   zorder=10, label="Start", marker="^")
        ax.scatter(*prob["x_star"], s=130, c="red", edgecolors="black",
                   zorder=10, label="x*", marker="*")

        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel("x₀"); ax.set_ylabel("x₁")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    plt.tight_layout()
    _savefig(fig, os.path.join(RESULTS_DIR, f"trajectory_{tag}.png"))


# ---------------------------------------------------------------------------
# Plot 4: Error ||xk - x*|| vs iterations — LOG x, LOG y
# ---------------------------------------------------------------------------

def plot_error_vs_iterations(prob, results, tag=""):
    """
    Distance to minimizer vs iterations on doubly-log axes.

    Two panels: NCG-only (cleaner view of variant differences) and
    all methods together (shows GD's slow progress clearly).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Error ‖xₖ − x*‖ vs. Iterations — {prob['name']}",
                 fontweight="bold", y=1.01)
    x_star = prob["x_star"]

    for ax, methods, title in zip(
        axes,
        [VARIANTS, ["GD", "scipy"] + VARIANTS],
        ["NCG Variants", "All Methods"]
    ):
        for m in methods:
            r = results[m]
            xh = _history(r, "x_history")
            if not xh:
                continue
            errors = np.maximum(
                [np.linalg.norm(np.asarray(x) - x_star) for x in xh], 1e-20)
            xs = _iters(errors)
            ax.semilogy(xs + 1, errors, **_style(m),
                        label=f"{m} ({_nit(r)} it)" if _converged(r)
                              else f"{m} (≥{_nit(r)})")
            _mark_unconverged(ax, r, xs[-1] + 1, errors[-1], COLORS[m])

        ax.set_title(title)
        ax.set_ylabel("‖xₖ − x*‖")
        _add_log_xaxis(ax)
        ax.legend(loc="upper right", framealpha=0.85)

    plt.tight_layout()
    _savefig(fig, os.path.join(RESULTS_DIR, f"error_vs_iter_{tag}.png"))


# ---------------------------------------------------------------------------
# Plot 5: Method comparison bar chart — LOG y
# ---------------------------------------------------------------------------

def plot_comparison_bar(all_results_by_prob):
    """
    Grouped bar chart: iterations to convergence for each method × problem.

    Log y-scale is essential because NCG converges in ~9 iterations on the
    quadratic bowl while GD takes 402 — a 45× difference that makes a linear
    y-axis compress NCG bars to invisibility.

    Methods that hit max_iter are shown at their cap value with a hatched
    pattern and a '▲ max' annotation to distinguish non-convergence clearly.
    """
    problems  = list(all_results_by_prob.keys())
    methods   = VARIANTS + ["GD", "scipy"]
    n_prob    = len(problems)
    n_meth    = len(methods)
    bar_w     = 0.10
    x_pos     = np.arange(n_prob)
    MAX_ITER  = 5000

    fig, ax = plt.subplots(figsize=(max(11, 2.2 * n_prob), 6))

    for i, m in enumerate(methods):
        iters   = []
        hit_max = []
        for p in problems:
            r = all_results_by_prob[p].get(m)
            if r is None:
                iters.append(1); hit_max.append(False)
            else:
                n = _nit(r)
                iters.append(n)
                hit_max.append(not _converged(r))

        offset = (i - n_meth / 2 + 0.5) * bar_w
        color  = COLORS[m]

        bars = ax.bar(x_pos + offset, iters, bar_w,
                      label=m, color=color, alpha=0.85,
                      edgecolor="white", linewidth=0.5, zorder=3)

        # Hatch bars that hit max_iter to signal non-convergence
        for bar, maxed in zip(bars, hit_max):
            if maxed:
                bar.set_hatch("///")
                bar.set_edgecolor("black")
                bar.set_linewidth(0.8)
                # Small triangle annotation
                ax.annotate("▲", xy=(bar.get_x() + bar.get_width()/2,
                                     bar.get_height()),
                            ha="center", va="bottom", fontsize=6,
                            color="black", zorder=6)

    ax.set_yscale("log")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([p[:28] for p in problems], rotation=22, ha="right")
    ax.set_ylabel("Iterations to Convergence (log scale)")
    ax.set_title("Method Comparison: Iterations to Convergence\n"
                 "(▲ = hit max-iter budget, did not converge)",
                 fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else f"{x:.1g}"))
    ax.legend(loc="upper right", framealpha=0.9, ncol=2)
    ax.grid(axis="y", which="both", ls=":", alpha=0.4)
    ax.grid(axis="y", which="major", ls=":", alpha=0.6)

    plt.tight_layout()
    _savefig(fig, os.path.join(RESULTS_DIR, "comparison_bar.png"))


# ---------------------------------------------------------------------------
# Plot 6: Beta and alpha diagnostics — LOG x, alpha LOG y
# ---------------------------------------------------------------------------

def plot_beta_and_alpha(prob, results, tag=""):
    """
    NCG diagnostic plots: beta values and step sizes per iteration.

    Log x-axis: consistent with all other iteration plots.
    Log y-axis on alpha: step sizes span many orders of magnitude on
    nonlinear problems (Wolfe line search can return very small alpha
    near saddle regions).
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    fig.suptitle(f"NCG Diagnostics — {prob['name']}", fontweight="bold")
    ax_beta, ax_alpha = axes

    for v in VARIANTS:
        r = results[v]
        if not isinstance(r, NCGResult) or not r.beta_history:
            continue
        xs = np.arange(1, len(r.beta_history) + 1)  # start from 1 for log scale
        ax_beta.semilogx(xs, r.beta_history, **_style(v), label=v)
        # Alpha: use log-log (step sizes can vary by orders of magnitude)
        ax_alpha.loglog(xs, np.maximum(r.alpha_history, 1e-20),
                        **_style(v), label=v)

    ax_beta.axhline(0, color="black", lw=0.9, ls="--", alpha=0.5)
    ax_beta.set_ylabel("βₖ")
    ax_beta.set_title("Beta (conjugacy coefficient)")
    ax_beta.legend(loc="upper right", framealpha=0.85)
    ax_beta.grid(True, which="both", ls=":", alpha=0.4)
    ax_beta.set_xlabel("Iteration (log scale)")

    ax_alpha.set_ylabel("αₖ  (step size, log scale)")
    ax_alpha.set_xlabel("Iteration (log scale)")
    ax_alpha.set_title("Step size (Wolfe line search)")
    ax_alpha.legend(loc="upper right", framealpha=0.85)
    ax_alpha.grid(True, which="both", ls=":", alpha=0.4)
    ax_alpha.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else f"{x:.1f}"))

    plt.tight_layout()
    _savefig(fig, os.path.join(RESULTS_DIR, f"beta_alpha_{tag}.png"))


# ---------------------------------------------------------------------------
# Plot 7: Finite-step termination — LOG-LOG
# ---------------------------------------------------------------------------

def plot_finite_termination():
    """
    Iterations to convergence vs condition number κ.

    Log-log axes: κ spans [2, 500] and iterations span [~10, >1000],
    both covering multiple orders of magnitude. Log-log makes the
    theoretical O(sqrt(κ)) and O(κ) scaling laws appear as straight lines.
    """
    kappas = [2, 5, 10, 20, 50, 100, 200, 500]
    n      = 20
    tol    = 1e-10

    ncg_iters = []; gd_iters = []

    for kappa in kappas:
        f, g, xs, fs = high_dim_quadratic(n=n, kappa=kappa, seed=0)
        x0 = np.zeros(n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res_ncg = ncg_minimize(f, g, x0, beta_variant="PR+", tol=tol, max_iter=5000)
            res_gd  = gradient_descent(f, g, x0, tol=tol, max_iter=50000)
        ncg_iters.append(res_ncg.nit)
        gd_iters.append(res_gd["nit"])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.loglog(kappas, ncg_iters, "o-", **_style("PR+"), label="NCG-PR+", markersize=7)
    ax.loglog(kappas, gd_iters,  "s--", **_style("GD"),  label="Gradient Descent", markersize=7)
    ax.axhline(n, color="gray", ls=":", lw=1.6, label=f"n = {n}  (CG bound)")

    # Reference lines: O(sqrt(kappa)) and O(kappa)
    kk = np.array(kappas, dtype=float)
    c1 = ncg_iters[0] / kk[0]**0.5
    c2 = gd_iters[0]  / kk[0]
    ax.loglog(kk, c1 * kk**0.5, "k:", lw=1, alpha=0.4, label=r"O($\sqrt{\kappa}$) ref")
    ax.loglog(kk, c2 * kk,      "k--", lw=1, alpha=0.4, label=r"O($\kappa$) ref")

    ax.set_xlabel("Condition Number κ(A)")
    ax.set_ylabel("Iterations to Convergence")
    ax.set_title(f"Finite Termination Study — {n}-D Quadratic\n"
                 f"(tol = {tol:.0e}, both axes log scale)", fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.grid(True, which="major", ls=":", alpha=0.6)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else f"{x:.1g}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else f"{x:.1g}"))

    plt.tight_layout()
    _savefig(fig, os.path.join(RESULTS_DIR, "finite_termination.png"))


# ---------------------------------------------------------------------------
# Plot 8: Four-panel convergence dashboard
# ---------------------------------------------------------------------------

def plot_dashboard(prob, results, tag=""):
    """
    Four-panel summary dashboard with consistent log-x scaling.

    Panel layout:
      TL: loss (log-log)        TR: gradient norm (log-log)
      BL: distance to x* (log-log)  BR: 2-D trajectory OR text
    """
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"NCG Convergence Dashboard — {prob['name']}",
                 fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)
    ax_loss  = fig.add_subplot(gs[0, 0])
    ax_gnorm = fig.add_subplot(gs[0, 1])
    ax_err   = fig.add_subplot(gs[1, 0])
    ax_traj  = fig.add_subplot(gs[1, 1])

    x_star = prob["x_star"]
    f_star = prob["f_star"]

    for m in VARIANTS + ["GD"]:
        r  = results[m]
        fh = _history(r, "fun_history")
        gh = _history(r, "grad_norm_history")
        xh = _history(r, "x_history")
        sty = _style(m)
        lbl = f"{m} ({_nit(r)} it)" if _converged(r) else f"{m} (≥{_nit(r)})"

        if fh:
            shifted = np.maximum(np.array(fh, dtype=float) - f_star, 1e-20)
            xs = _iters(shifted) + 1
            ax_loss.semilogy(xs, shifted, **sty, label=lbl)
            _mark_unconverged(ax_loss, r, xs[-1], shifted[-1], COLORS[m])

        if gh:
            xs = _iters(gh) + 1
            ax_gnorm.semilogy(xs, gh, **sty, label=lbl)
            _mark_unconverged(ax_gnorm, r, xs[-1], gh[-1], COLORS[m])

        if xh:
            errs = np.maximum(
                [np.linalg.norm(np.asarray(x) - x_star) for x in xh], 1e-20)
            xs = _iters(errs) + 1
            ax_err.semilogy(xs, errs, **sty, label=lbl)
            _mark_unconverged(ax_err, r, xs[-1], errs[-1], COLORS[m])

    for ax, ylabel, title in [
        (ax_loss,  "f(xₖ) − f*",  "Loss (log-log)"),
        (ax_gnorm, "‖∇f(xₖ)‖",   "Gradient Norm (log-log)"),
        (ax_err,   "‖xₖ − x*‖",  "Distance to Minimizer (log-log)"),
    ]:
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
        _add_log_xaxis(ax)

    # Trajectory panel (2-D only)
    if prob["dim"] == 2:
        f = prob["f"]
        all_xh = []
        for r in results.values():
            xh = _history(r, "x_history")
            if xh:
                all_xh.extend(xh)
        if all_xh:
            pts = np.array(all_xh)
            pad = 0.5
            xlim = (pts[:,0].min()-pad, pts[:,0].max()+pad)
            ylim = (pts[:,1].min()-pad, pts[:,1].max()+pad)
            gx = np.linspace(xlim[0], xlim[1], 200)
            gy = np.linspace(ylim[0], ylim[1], 200)
            X2, Y2 = np.meshgrid(gx, gy)
            Z = np.vectorize(lambda x, y: f(np.array([x, y])))(X2, Y2)
            ax_traj.contourf(X2, Y2, np.log1p(Z - Z.min()), levels=40,
                             cmap="viridis", alpha=0.65)
            ax_traj.contour(X2, Y2, np.log1p(Z - Z.min()), levels=20,
                            colors="white", linewidths=0.3, alpha=0.35)

            for m in ["GD", "PR+"]:
                r  = results[m]
                xh = _history(r, "x_history")
                if xh:
                    path = np.array(xh)
                    ax_traj.plot(path[:,0], path[:,1], "o-",
                                 color=COLORS[m], markersize=2, lw=1.5,
                                 label=f"{m} ({len(xh)-1} steps)", alpha=0.88)

            ax_traj.scatter(*prob["x0"], s=110, c="white",
                            edgecolors="black", zorder=10, label="Start", marker="^")
            ax_traj.scatter(*x_star, s=110, c="red",
                            edgecolors="black", zorder=10, label="x*",  marker="*")
            ax_traj.set_xlim(xlim); ax_traj.set_ylim(ylim)
            ax_traj.set_xlabel("x₀"); ax_traj.set_ylabel("x₁")
            ax_traj.set_title("2-D Trajectory (PR+ vs GD)")
            ax_traj.legend(fontsize=8, framealpha=0.85)
    else:
        ax_traj.axis("off")
        ax_traj.text(0.5, 0.5,
                     "Trajectory plot\nnot available for\nhigh-dim problems",
                     ha="center", va="center", transform=ax_traj.transAxes,
                     fontsize=12, color="gray")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    _savefig(fig, os.path.join(RESULTS_DIR, f"dashboard_{tag}.png"))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_all_plots():
    print("\n" + "="*60)
    print("  GENERATING ALL NCG CONVERGENCE PLOTS")
    print("="*60)

    problems = get_all_test_problems()
    tags     = ["quadratic_2d", "rosenbrock", "beale", "highdim_quad"]
    all_results_by_prob = {}

    for prob, tag in zip(problems, tags):
        print(f"\n--- {prob['name']} ---")
        results = _run_all(prob, tol=1e-8)
        all_results_by_prob[prob["name"]] = results

        plot_loss_vs_iterations(prob, results, tag)
        plot_gradient_norm(prob, results, tag, run_tol=1e-8)
        plot_error_vs_iterations(prob, results, tag)
        plot_beta_and_alpha(prob, results, tag)
        plot_dashboard(prob, results, tag)

        if prob["dim"] == 2:
            plot_2d_trajectory(prob, results, tag)

    print("\n--- Comparison Bar Chart ---")
    plot_comparison_bar(all_results_by_prob)

    print("\n--- Finite Termination Study ---")
    plot_finite_termination()

    print(f"\n✓ All plots saved to: {os.path.abspath(RESULTS_DIR)}")


if __name__ == "__main__":
    generate_all_plots()
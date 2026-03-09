"""
plots.py — Convergence Demonstration Plots for NCG
===================================================
Generates all figures described in the To-Do Checklist §7:

  1. Loss (function value) vs. iterations — log scale
  2. Gradient norm ‖∇f(xk)‖ vs. iterations — log scale
  3. 2-D trajectory on contour plot (quadratic bowl & Rosenbrock)
  4. Error ‖xk - x*‖ vs. iterations — log scale
  5. Method comparison: NCG variants vs. GD vs. scipy on each problem
  6. Beta values and step sizes over iterations
  7. Condition-number study (finite-step termination on quadratics)

All figures are saved to results/ directory.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

# ---- local imports ---------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from ncg import ncg_minimize, NCGResult
from test_functions import (quadratic_bowl_2d, rosenbrock, beale,
                             high_dim_quadratic, get_all_test_problems)
from comparison import gradient_descent, run_scipy

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

VARIANTS = ["FR", "PR", "PR+", "HS"]
COLORS   = {"FR": "#e63946", "PR": "#457b9d", "PR+": "#2a9d8f",
            "HS": "#e9c46a", "GD": "#6d6875", "scipy": "#264653"}
LINESTYLES = {"FR": "-", "PR": "--", "PR+": "-.", "HS": ":",
              "GD": (0,(3,1,1,1)), "scipy": "-"}

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi":   120,
})


# ---------------------------------------------------------------------------
# Helper: run one problem with all NCG variants + GD
# ---------------------------------------------------------------------------

def _run_all(prob, tol=1e-8):
    results = {}
    for v in VARIANTS:
        results[v] = ncg_minimize(prob["f"], prob["grad_f"],
                                  prob["x0"].copy(),
                                  beta_variant=v, tol=tol, max_iter=5000)
    results["GD"] = gradient_descent(prob["f"], prob["grad_f"],
                                     prob["x0"].copy(), tol=tol, max_iter=5000)
    results["scipy"] = run_scipy(prob["f"], prob["grad_f"],
                                 prob["x0"].copy(), tol=tol)
    return results


# ---------------------------------------------------------------------------
# Plot 1: Loss vs iterations (log scale)
# ---------------------------------------------------------------------------

def plot_loss_vs_iterations(prob, results, tag=""):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    fig.suptitle(f"Loss vs. Iterations — {prob['name']}", fontweight="bold")

    for ax, methods, title in zip(
        axes,
        [VARIANTS, ["GD", "scipy"] + VARIANTS],
        ["NCG Variants", "All Methods"]
    ):
        for m in methods:
            r = results[m]
            fh = r.fun_history if isinstance(r, NCGResult) else r.get("fun_history", [])
            if not fh:
                continue
            f_min = prob["f_star"]
            # Shift so we plot f(xk) - f* on log scale
            shifted = np.maximum(np.array(fh) - f_min, 1e-20)
            ax.semilogy(shifted,
                        color=COLORS[m], linestyle=LINESTYLES[m],
                        linewidth=1.8, label=m, alpha=0.85)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("f(xₖ) − f*")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(True, which="both", ls=":", alpha=0.5)

    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"loss_vs_iter_{tag}.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Plot 2: Gradient norm vs iterations (log scale)
# ---------------------------------------------------------------------------

def plot_gradient_norm(prob, results, tag="", run_tol=1e-8):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.suptitle(f"Gradient Norm vs. Iterations — {prob['name']}",
                 fontweight="bold")

    for m in VARIANTS + ["GD"]:
        r = results[m]
        gh = r.grad_norm_history if isinstance(r, NCGResult) else r.get("grad_norm_history", [])
        if not gh:
            continue
        ax.semilogy(gh, color=COLORS[m], linestyle=LINESTYLES[m],
                    linewidth=1.8, label=m, alpha=0.85)

    ax.axhline(run_tol, color="black", ls=":", lw=1.2, label=f"tol={run_tol:.0e}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("‖∇f(xₖ)‖")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", ls=":", alpha=0.5)

    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"grad_norm_{tag}.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Plot 3: 2-D trajectory on contour plot
# ---------------------------------------------------------------------------

def plot_2d_trajectory(prob, results, tag="", n_contours=30,
                       xlim=None, ylim=None):
    """
    Overlays optimisation trajectories on the contour plot of the objective.
    Only meaningful for 2-D problems.
    """
    if prob["dim"] != 2:
        return

    f = prob["f"]

    # --- Build grid for contours ---
    x0_all = [r.x_history if isinstance(r, NCGResult) else r.get("x_history", [])
               for r in results.values() if (isinstance(r, NCGResult) and r.x_history) or
               (isinstance(r, dict) and r.get("x_history"))]

    all_pts = np.concatenate([np.array(xh) for xh in x0_all if xh], axis=0)
    x0_start = prob["x0"]

    if xlim is None:
        pad = 0.5
        xlim = (min(all_pts[:,0].min(), x0_start[0]) - pad,
                max(all_pts[:,0].max(), x0_start[0]) + pad)
    if ylim is None:
        pad = 0.5
        ylim = (min(all_pts[:,1].min(), x0_start[1]) - pad,
                max(all_pts[:,1].max(), x0_start[1]) + pad)

    gx = np.linspace(xlim[0], xlim[1], 300)
    gy = np.linspace(ylim[0], ylim[1], 300)
    X, Y = np.meshgrid(gx, gy)
    Z = np.vectorize(lambda x, y: f(np.array([x, y])))(X, Y)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"2-D Trajectory — {prob['name']}", fontweight="bold")

    for ax, method_group, title in zip(
        axes,
        [["FR", "PR", "PR+", "HS"], ["GD", "PR+"]],
        ["NCG Variants", "NCG-PR+ vs Gradient Descent"]
    ):
        cs = ax.contourf(X, Y, np.log1p(Z - Z.min()), levels=40,
                         cmap="viridis", alpha=0.6)
        ax.contour(X, Y, np.log1p(Z - Z.min()), levels=40,
                   colors="white", linewidths=0.4, alpha=0.4)
        plt.colorbar(cs, ax=ax, shrink=0.85, label="log(1 + f - fmin)")

        for m in method_group:
            r = results[m]
            xh = r.x_history if isinstance(r, NCGResult) else r.get("x_history", [])
            if not xh:
                continue
            path = np.array(xh)
            ax.plot(path[:,0], path[:,1], "o-",
                    color=COLORS[m], markersize=2.5,
                    linewidth=1.5, label=f"{m} ({len(xh)-1} iter)",
                    alpha=0.85, zorder=5)

        # Mark start and minimizer
        ax.scatter(*x0_start, s=120, c="white", edgecolors="black",
                   zorder=10, label="Start", marker="^")
        ax.scatter(*prob["x_star"], s=120, c="red", edgecolors="black",
                   zorder=10, label="x*", marker="*")

        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel("x₀"); ax.set_ylabel("x₁")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"trajectory_{tag}.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Plot 4: Error ‖xk - x*‖ vs iterations (log scale)
# ---------------------------------------------------------------------------

def plot_error_vs_iterations(prob, results, tag=""):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.suptitle(f"Error ‖xₖ − x*‖ vs. Iterations — {prob['name']}",
                 fontweight="bold")

    x_star = prob["x_star"]

    for m in VARIANTS + ["GD"]:
        r = results[m]
        xh = r.x_history if isinstance(r, NCGResult) else r.get("x_history", [])
        if not xh:
            continue
        errors = np.maximum([np.linalg.norm(np.array(x) - x_star) for x in xh], 1e-20)
        ax.semilogy(errors, color=COLORS[m], linestyle=LINESTYLES[m],
                    linewidth=1.8, label=m, alpha=0.85)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("‖xₖ − x*‖")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", ls=":", alpha=0.5)

    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"error_vs_iter_{tag}.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Plot 5: Comparison bar chart (iterations to convergence)
# ---------------------------------------------------------------------------

def plot_comparison_bar(all_results_by_prob):
    """
    Grouped bar chart: iterations to convergence for each method × problem.
    """
    problems   = list(all_results_by_prob.keys())
    methods    = VARIANTS + ["GD", "scipy"]
    n_prob     = len(problems)
    n_meth     = len(methods)
    bar_width  = 0.1
    x_pos      = np.arange(n_prob)

    fig, ax = plt.subplots(figsize=(max(10, 2*n_prob), 5))

    for i, m in enumerate(methods):
        iters = []
        for p in problems:
            r = all_results_by_prob[p].get(m)
            if r is None:
                iters.append(0)
            elif isinstance(r, NCGResult):
                iters.append(r.nit)
            else:
                iters.append(r.get("nit", 0))
        offset = (i - n_meth / 2 + 0.5) * bar_width
        color_key = m if m in COLORS else ("scipy" if "scipy" in m else "GD")
        ax.bar(x_pos + offset, iters, bar_width,
               label=m, color=COLORS.get(color_key, "#888"),
               alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([p[:25] for p in problems], rotation=20, ha="right")
    ax.set_ylabel("Iterations to Convergence")
    ax.set_title("Method Comparison: Iterations to Convergence", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", ls=":", alpha=0.5)

    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, "comparison_bar.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Plot 6: Beta values and step sizes over iterations (NCG diagnostics)
# ---------------------------------------------------------------------------

def plot_beta_and_alpha(prob, results, tag=""):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    fig.suptitle(f"NCG Diagnostics — {prob['name']}", fontweight="bold")

    ax_beta, ax_alpha = axes

    for v in VARIANTS:
        r = results[v]
        if not isinstance(r, NCGResult) or not r.beta_history:
            continue
        ax_beta.plot(r.beta_history,  color=COLORS[v], linestyle=LINESTYLES[v],
                     linewidth=1.5, label=v, alpha=0.85)
        ax_alpha.plot(r.alpha_history, color=COLORS[v], linestyle=LINESTYLES[v],
                      linewidth=1.5, label=v, alpha=0.85)

    ax_beta.axhline(0, color="black", lw=0.8, ls="--")
    ax_beta.set_ylabel("βₖ")
    ax_beta.set_title("Beta (conjugacy coefficient) per iteration")
    ax_beta.legend(loc="upper right")
    ax_beta.grid(True, ls=":", alpha=0.5)

    ax_alpha.set_ylabel("αₖ  (step size)")
    ax_alpha.set_xlabel("Iteration")
    ax_alpha.set_title("Step size (Wolfe line search) per iteration")
    ax_alpha.legend(loc="upper right")
    ax_alpha.grid(True, ls=":", alpha=0.5)
    ax_alpha.set_yscale("log")

    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"beta_alpha_{tag}.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Plot 7: Finite-step termination on quadratics — condition number study
# ---------------------------------------------------------------------------

def plot_finite_termination():
    """
    Demonstrates NCG's finite-step termination on quadratic functions.
    Shows iterations to convergence vs. condition number for NCG-PR+ vs. GD.
    """
    kappas = [2, 5, 10, 20, 50, 100, 200, 500]
    n = 20
    tol = 1e-10

    ncg_iters = []
    gd_iters  = []

    for kappa in kappas:
        from test_functions import high_dim_quadratic
        f, g, xs, fs = high_dim_quadratic(n=n, kappa=kappa, seed=0)
        x0 = np.zeros(n)

        res_ncg = ncg_minimize(f, g, x0, beta_variant="PR+", tol=tol, max_iter=5000)
        res_gd  = gradient_descent(f, g, x0, tol=tol, max_iter=50000)
        ncg_iters.append(res_ncg.nit)
        gd_iters.append(res_gd["nit"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(kappas, ncg_iters, "o-", color=COLORS["PR+"], lw=2, label="NCG-PR+")
    ax.plot(kappas, gd_iters,  "s--", color=COLORS["GD"],  lw=2, label="Gradient Descent")
    ax.axhline(n, color="gray", ls=":", lw=1.5, label=f"n = {n}  (CG bound)")
    ax.set_xscale("log")
    ax.set_xlabel("Condition Number κ(A)")
    ax.set_ylabel("Iterations to Convergence")
    ax.set_title(f"Finite Termination Study — {n}-D Quadratic", fontweight="bold")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.5)

    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, "finite_termination.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Plot 8: Four-panel convergence dashboard per problem
# ---------------------------------------------------------------------------

def plot_dashboard(prob, results, tag=""):
    """
    Four-panel figure combining all convergence metrics in one view:
    Top-left: loss, Top-right: gradient norm,
    Bottom-left: error, Bottom-right: 2D trajectory (if 2D)
    """
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"NCG Convergence Dashboard — {prob['name']}",
                 fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)
    ax_loss  = fig.add_subplot(gs[0, 0])
    ax_gnorm = fig.add_subplot(gs[0, 1])
    ax_err   = fig.add_subplot(gs[1, 0])
    ax_traj  = fig.add_subplot(gs[1, 1])

    x_star = prob["x_star"]
    f_star = prob["f_star"]

    for m in VARIANTS + ["GD"]:
        r = results[m]
        if isinstance(r, NCGResult):
            fh = r.fun_history
            gh = r.grad_norm_history
            xh = r.x_history
        else:
            fh = r.get("fun_history", [])
            gh = r.get("grad_norm_history", [])
            xh = r.get("x_history", [])

        c = COLORS[m]; ls = LINESTYLES[m]

        if fh:
            shifted = np.maximum(np.array(fh) - f_star, 1e-20)
            ax_loss.semilogy(shifted, color=c, linestyle=ls, lw=1.8,
                             label=m, alpha=0.85)
        if gh:
            ax_gnorm.semilogy(gh, color=c, linestyle=ls, lw=1.8,
                              label=m, alpha=0.85)
        if xh:
            errors = np.maximum(
                [np.linalg.norm(np.array(x) - x_star) for x in xh], 1e-20)
            ax_err.semilogy(errors, color=c, linestyle=ls, lw=1.8,
                            label=m, alpha=0.85)

    for ax, ylabel, title in [
        (ax_loss,  "f(xₖ) − f*",    "Loss (log scale)"),
        (ax_gnorm, "‖∇f(xₖ)‖",      "Gradient Norm"),
        (ax_err,   "‖xₖ − x*‖",     "Distance to Minimizer"),
    ]:
        ax.set_xlabel("Iteration"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, which="both", ls=":", alpha=0.5)

    # Trajectory panel (2D only)
    if prob["dim"] == 2:
        f = prob["f"]
        all_xh = []
        for r in results.values():
            xh = r.x_history if isinstance(r, NCGResult) else r.get("x_history",[])
            if xh:
                all_xh.extend(xh)
        if all_xh:
            pts = np.array(all_xh)
            pad = 0.5
            xlim = (pts[:,0].min()-pad, pts[:,0].max()+pad)
            ylim = (pts[:,1].min()-pad, pts[:,1].max()+pad)
            gx = np.linspace(xlim[0], xlim[1], 200)
            gy = np.linspace(ylim[0], ylim[1], 200)
            X, Y = np.meshgrid(gx, gy)
            Z = np.vectorize(lambda x, y: f(np.array([x, y])))(X, Y)
            ax_traj.contourf(X, Y, np.log1p(Z - Z.min()), levels=40,
                             cmap="viridis", alpha=0.65)
            ax_traj.contour(X, Y, np.log1p(Z - Z.min()), levels=20,
                            colors="white", linewidths=0.3, alpha=0.4)

            for m in ["GD", "PR+"]:
                r = results[m]
                xh = r.x_history if isinstance(r, NCGResult) else r.get("x_history",[])
                if xh:
                    path = np.array(xh)
                    ax_traj.plot(path[:,0], path[:,1], "o-",
                                 color=COLORS[m], markersize=2, lw=1.5,
                                 label=m, alpha=0.85)

            ax_traj.scatter(*prob["x0"], s=100, c="white",
                            edgecolors="black", zorder=10, label="start", marker="^")
            ax_traj.scatter(*x_star, s=100, c="red",
                            edgecolors="black", zorder=10, label="x*", marker="*")
            ax_traj.set_xlim(xlim); ax_traj.set_ylim(ylim)
            ax_traj.set_xlabel("x₀"); ax_traj.set_ylabel("x₁")
            ax_traj.set_title("2-D Trajectory (PR+ vs GD)")
            ax_traj.legend(fontsize=8)
    else:
        ax_traj.axis("off")
        ax_traj.text(0.5, 0.5, "(Trajectory plot\nnot available for\nhigh-dim problems)",
                     ha="center", va="center", transform=ax_traj.transAxes,
                     fontsize=12, color="gray")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"dashboard_{tag}.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_all_plots():
    """Run all benchmark tests and generate every plot in the checklist."""
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

    # Cross-problem comparison
    print("\n--- Comparison Bar Chart ---")
    plot_comparison_bar(all_results_by_prob)

    # Finite termination study
    print("\n--- Finite Termination Study ---")
    plot_finite_termination()

    print(f"\n✓ All plots saved to: {os.path.abspath(RESULTS_DIR)}")


if __name__ == "__main__":
    generate_all_plots()
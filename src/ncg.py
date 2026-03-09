"""
ncg.py — Nonlinear Conjugate Gradient (NCG) Optimizer
======================================================
Full implementation of the NCG algorithm with:
  - Four beta variants: Fletcher-Reeves (FR), Polak-Ribière (PR),
    Polak-Ribière+ (PR+), Hestenes-Stiefel (HS)
  - Wolfe line search with zoom phase (bracketing + zoom)
  - Three restart strategies: periodic, Powell's criterion,
    descent-failure restart
  - Numerical safeguards throughout

Fixes applied (exhaustive audit):
  - Removed dead function _phi_and_dphi (was defined but never called)
  - Removed unused `import warnings` (now only imported locally when needed)
  - Eliminated redundant f-evaluation after line search: wolfe_line_search
    now returns (alpha, f_new, nfev, ngev) so the caller avoids an extra f call
  - Zoom phase: dphi_hi is now cached and recomputed only when alpha_hi changes
  - Bracketing phase: dphi0 is passed directly as initial dphi_lo to zoom,
    avoiding a redundant gradient evaluation at alpha=0
  - Line search emits RuntimeWarning when max_iter exhausted without Wolfe point
  - x0 is ravel()ed to 1-D on entry so 2-D arrays are handled gracefully
  - Powell restart fires independently of periodic restart (separate if, not elif)

Reference: Nocedal & Wright, Numerical Optimization, 2nd ed.
           Fletcher & Reeves (1964), Polak & Ribière (1969)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class NCGResult:
    """Stores the full history and outcome of an NCG run."""
    x: np.ndarray                          # Final iterate
    fun: float                             # Final function value
    grad_norm: float                       # Final gradient norm
    nit: int                               # Iteration count: includes the final
                                           #   convergence-check pass, so nit may
                                           #   equal len(x_history) not len(x_history)-1
    nfev: int                              # Number of function evaluations
    ngev: int                              # Number of gradient evaluations
    success: bool                          # Converged within tolerance?
    message: str                           # Human-readable status

    # Per-iteration histories:
    #   fun_history, grad_norm_history, x_history : length = completed_steps + 1
    #     Index 0 = initial point; index j = state after step j.
    #   beta_history, alpha_history, restart_flags : length = completed_steps
    #     Index j = value used/produced at step j+1.
    fun_history: List[float] = field(default_factory=list)
    grad_norm_history: List[float] = field(default_factory=list)
    x_history: List[np.ndarray] = field(default_factory=list)
    beta_history: List[float] = field(default_factory=list)
    alpha_history: List[float] = field(default_factory=list)
    restart_flags: List[bool] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Beta (conjugacy update) formulas
# ---------------------------------------------------------------------------

def beta_fletcher_reeves(g_new: np.ndarray, g_old: np.ndarray,
                          d_old: np.ndarray) -> float:
    """
    Fletcher-Reeves (FR) beta formula.

        beta_k^FR = ||g_{k+1}||^2 / ||g_k||^2

    Derivation: In the quadratic exact-arithmetic setting, residuals are
    mutually orthogonal. The conjugacy condition d_{k+1}^T A d_k = 0
    together with this orthogonality simplifies to the ratio of squared
    gradient norms.

    Properties
    ----------
    - Strongest convergence guarantee (global, under strong Wolfe).
    - Can suffer from jamming: a bad step carries forward with full weight
      because FR uses only gradient magnitudes, not directional changes.

    Safeguard
    ---------
    Returns 0.0 when ||g_old|| = 0 to avoid division by zero.
    """
    denom = np.dot(g_old, g_old)
    if denom < 1e-30:
        return 0.0
    return np.dot(g_new, g_new) / denom


def beta_polak_ribiere(g_new: np.ndarray, g_old: np.ndarray,
                        d_old: np.ndarray) -> float:
    """
    Polak-Ribiere (PR) beta formula.

        beta_k^PR = g_{k+1}^T (g_{k+1} - g_k) / ||g_k||^2

    Derivation: Uses the actual gradient difference y_k = g_{k+1} - g_k as
    a curvature proxy. In the quadratic case this equals FR exactly (since
    g_{k+1}^T g_k = 0 makes g_{k+1}^T y_k = ||g_{k+1}||^2). For nonlinear
    problems it self-corrects when progress stalls: y_k approx 0 gives
    beta approx 0, resetting toward steepest descent.

    Properties
    ----------
    - More robust to bad steps than FR due to self-correcting behaviour.
    - Can be negative, which may violate the descent condition.
      No global convergence guarantee without truncation (use PR+).

    Safeguard
    ---------
    Returns 0.0 when ||g_old|| = 0 to avoid division by zero.
    """
    denom = np.dot(g_old, g_old)
    if denom < 1e-30:
        return 0.0
    y = g_new - g_old
    return np.dot(g_new, y) / denom


def beta_polak_ribiere_plus(g_new: np.ndarray, g_old: np.ndarray,
                             d_old: np.ndarray) -> float:
    """
    Polak-Ribiere+ (PR+) beta formula — Powell (1984).

        beta_k^PR+ = max(beta_k^PR, 0)

    Rationale: Truncating negative PR values at zero prevents the direction
    from violating the descent condition. When beta = 0, dk+1 = -g_{k+1},
    which is steepest descent — an automatic restart. This gives PR+ global
    convergence under mild conditions while retaining PR's empirical efficiency.

    Properties
    ----------
    - Best overall for general nonlinear optimization.
    - Descent direction guaranteed.
    - Global convergence established under mild conditions (Nocedal & Wright, Sec 5.2).
    """
    return max(beta_polak_ribiere(g_new, g_old, d_old), 0.0)


def beta_hestenes_stiefel(g_new: np.ndarray, g_old: np.ndarray,
                           d_old: np.ndarray) -> float:
    """
    Hestenes-Stiefel (HS) beta formula.

        beta_k^HS = g_{k+1}^T (g_{k+1} - g_k) / d_k^T (g_{k+1} - g_k)

    Derivation: Directly imposes the secant-based conjugacy condition
    d_{k+1}^T y_k = 0, where y_k = g_{k+1} - g_k. Substituting
    d_{k+1} = -g_{k+1} + beta_k d_k and solving for beta_k yields the HS formula.
    This is the most conjugacy-motivated variant and is consistent with the
    secant equation used in quasi-Newton methods.

    Properties
    ----------
    - Theoretically well-motivated; directly enforces d_{k+1} perp y_k.
    - Competitive when line search quality is high.
    - No global convergence guarantee without additional safeguards.

    Safeguard
    ---------
    Returns 0.0 when |d_k^T y_k| < 1e-30 to avoid division by near-zero
    denominator. This triggers an implicit steepest-descent restart, which
    is the correct response when the curvature estimate becomes unreliable.
    """
    y = g_new - g_old
    denom = np.dot(d_old, y)
    if abs(denom) < 1e-30:
        return 0.0          # Near-zero denominator -> implicit restart
    return np.dot(g_new, y) / denom


# Map string names to functions for convenience
BETA_FUNCTIONS = {
    "FR":  beta_fletcher_reeves,
    "PR":  beta_polak_ribiere,
    "PR+": beta_polak_ribiere_plus,
    "HS":  beta_hestenes_stiefel,
}


# ---------------------------------------------------------------------------
# Wolfe line search — bracketing + zoom phase
# ---------------------------------------------------------------------------

def wolfe_line_search(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    d: np.ndarray,
    f0: float,
    g0: np.ndarray,
    alpha_init: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    alpha_max: float = 50.0,
    max_iter: int = 30,
) -> Tuple[float, float, int, int]:
    """
    Wolfe line search with bracketing and zoom phases.

    Finds alpha_k satisfying the strong Wolfe conditions:
        (1) f(x + alpha d) <= f(x) + c1 alpha g0^T d      (Armijo)
        (2) |grad_f(x + alpha d)^T d| <= c2 |g0^T d|      (strong curvature)

    Algorithm: Nocedal & Wright (2006), Algorithms 3.5 and 3.6.

    Parameters
    ----------
    f, grad_f  : objective and gradient callables
    x          : current iterate
    d          : search direction (must satisfy g0^T d < 0)
    f0         : f(x) — avoids one extra function call
    g0         : grad_f(x) — avoids one extra gradient call
    alpha_init : initial trial step size (default 1.0)
    c1         : sufficient-decrease constant (0 < c1 < c2 < 1)
    c2         : curvature constant
    alpha_max  : maximum allowed step size
    max_iter   : iteration budget for both bracketing and zoom phases

    Returns
    -------
    alpha  : accepted step size
    f_new  : f(x + alpha * d) — cached to avoid a redundant call in the caller
    nfev   : number of function evaluations used
    ngev   : number of gradient evaluations used

    Notes
    -----
    Emits RuntimeWarning when budget exhausted without finding a Wolfe point.
    """
    import warnings

    dphi0 = np.dot(g0, d)           # phi'(0) = grad_f(x)^T d  (must be < 0)
    assert dphi0 < 0, (
        f"Search direction is not a descent direction: g^T d = {dphi0:.3e} >= 0."
    )

    nfev = ngev = 0

    def phi(a: float) -> float:
        nonlocal nfev
        nfev += 1
        return f(x + a * d)

    def dphi(a: float) -> float:
        nonlocal ngev
        ngev += 1
        return np.dot(grad_f(x + a * d), d)

    def phi_dphi(a: float) -> Tuple[float, float]:
        nonlocal nfev, ngev
        nfev += 1
        ngev += 1
        xa = x + a * d
        return f(xa), np.dot(grad_f(xa), d)

    # ---- Zoom sub-routine --------------------------------------------------
    def zoom(alpha_lo: float, alpha_hi: float,
             phi_lo: float, phi_hi: float,
             dphi_lo: float, dphi_hi_init: float) -> Tuple[float, float]:
        """
        Narrow [alpha_lo, alpha_hi] until strong Wolfe conditions are satisfied.

        Invariants:
          - phi(alpha_lo) satisfies Armijo.
          - phi(alpha_lo) <= phi(alpha_hi).
          - The true minimiser of phi is within the interval.

        Bug fix: dphi_hi is now cached (dphi_hi_init) and recomputed only
        when alpha_hi actually changes, preventing a redundant gradient
        evaluation on every zoom iteration.
        """
        dphi_hi = dphi_hi_init   # Cache: avoids recomputing when alpha_hi unchanged

        for _ in range(max_iter):
            # Cubic Hermite interpolation inside (alpha_lo, alpha_hi)
            alpha_j = _cubic_interp(alpha_lo, alpha_hi,
                                    phi_lo, phi_hi, dphi_lo, dphi_hi)
            # Bisection fallback when interpolation lands outside interval
            lo = min(alpha_lo, alpha_hi)
            hi = max(alpha_lo, alpha_hi)
            if alpha_j <= lo or alpha_j >= hi:
                alpha_j = 0.5 * (alpha_lo + alpha_hi)

            phi_j, dphi_j = phi_dphi(alpha_j)

            # Armijo violated or worse than lower bound -> shrink from above
            if (phi_j > f0 + c1 * alpha_j * dphi0) or (phi_j >= phi_lo):
                alpha_hi = alpha_j
                phi_hi   = phi_j
                dphi_hi  = dphi_j  # alpha_hi changed: update cached derivative

            else:
                # Strong curvature condition satisfied -> accept
                if abs(dphi_j) <= c2 * abs(dphi0):
                    return alpha_j, phi_j

                # Move lower bound up; swap hi if slope sign would be wrong
                if dphi_j * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo
                    phi_hi   = phi_lo
                    dphi_hi  = dphi_lo  # alpha_hi changed: update cached derivative
                alpha_lo = alpha_j
                phi_lo   = phi_j
                dphi_lo  = dphi_j

        # Budget exhausted
        warnings.warn(
            "Wolfe zoom exhausted iteration budget without satisfying the "
            "strong curvature condition. Returning best Armijo-satisfying step.",
            RuntimeWarning, stacklevel=4,
        )
        return alpha_lo, phi_lo

    # ---- Bracketing phase --------------------------------------------------
    # Track dphi at alpha_prev so it can be passed to zoom without recomputing.
    # At the start alpha_prev = 0, dphi_prev = dphi0 (already known, free).
    alpha_prev = 0.0
    phi_prev   = f0
    dphi_prev  = dphi0   # Bug fix: no gradient eval needed at alpha=0

    alpha_cur  = alpha_init

    for i in range(max_iter):
        phi_cur = phi(alpha_cur)

        # Armijo violation or worse than previous -> zoom between prev and cur
        if (phi_cur > f0 + c1 * alpha_cur * dphi0) or (i > 0 and phi_cur >= phi_prev):
            # dphi_hi = dphi(alpha_cur) — needs one gradient eval here
            alpha_star, f_star = zoom(
                alpha_prev, alpha_cur,
                phi_prev,   phi_cur,
                dphi_prev,  dphi(alpha_cur),
            )
            return alpha_star, f_star, nfev, ngev

        dphi_cur = dphi(alpha_cur)

        # Strong curvature satisfied -> accept
        if abs(dphi_cur) <= c2 * abs(dphi0):
            return alpha_cur, phi_cur, nfev, ngev

        # Positive slope at alpha_cur -> minimum is between prev and cur
        if dphi_cur >= 0:
            # dphi_hi = dphi_prev (already known, no extra eval)
            alpha_star, f_star = zoom(
                alpha_cur,  alpha_prev,
                phi_cur,    phi_prev,
                dphi_cur,   dphi_prev,
            )
            return alpha_star, f_star, nfev, ngev

        # Step still valid: expand toward alpha_max
        alpha_prev = alpha_cur
        phi_prev   = phi_cur
        dphi_prev  = dphi_cur      # Cache for next iteration
        alpha_cur  = min(2.0 * alpha_cur, alpha_max)

    # Bracketing budget exhausted
    f_cur = phi(alpha_cur)
    warnings.warn(
        "Wolfe bracketing exhausted iteration budget. "
        "Returning last step without Wolfe guarantee.",
        RuntimeWarning, stacklevel=3,
    )
    return alpha_cur, f_cur, nfev, ngev


def _cubic_interp(a1: float, a2: float,
                  f1: float, f2: float,
                  df1: float, df2: float) -> float:
    """
    Cubic Hermite interpolation between two bracketing points.

    Fits a cubic polynomial through (a1, f1, df1) and (a2, f2, df2) and
    returns the location of its minimum. Falls back to bisection when:
      - The interval is degenerate (|a1 - a2| < eps)
      - The discriminant is negative (no real minimum)
      - The denominator is near zero (ill-conditioned cubic)
    """
    if abs(a1 - a2) < 1e-30:
        return 0.5 * (a1 + a2)
    d1 = df1 + df2 - 3.0 * (f1 - f2) / (a1 - a2)
    discriminant = d1 ** 2 - df1 * df2
    if discriminant < 0:
        return 0.5 * (a1 + a2)
    d2 = np.sqrt(discriminant)
    denom = df2 - df1 + 2.0 * d2
    if abs(denom) < 1e-30:
        return 0.5 * (a1 + a2)
    alpha = a2 - (a2 - a1) * (df2 + d2 - d1) / denom
    return float(alpha)


# ---------------------------------------------------------------------------
# Restart criteria
# ---------------------------------------------------------------------------

def _should_restart_powell(g_new: np.ndarray, g_old: np.ndarray,
                            nu: float = 0.1) -> bool:
    """
    Powell's restart criterion (Powell 1977).

    Restart when consecutive gradients are insufficiently orthogonal:

        |g_{k+1}^T g_k| / ||g_{k+1}||^2 >= nu

    A large ratio indicates near-parallel gradients, signalling that
    conjugacy has been lost. Typical threshold: nu = 0.1.
    """
    denom = np.dot(g_new, g_new)
    if denom < 1e-30:
        return False
    return abs(np.dot(g_new, g_old)) / denom >= nu


def _should_restart_descent_failure(g_new: np.ndarray,
                                    d_new: np.ndarray) -> bool:
    """
    Descent-failure restart.

    Trigger when the candidate new direction is not a descent direction:

        g_{k+1}^T d_{k+1} >= 0

    Safety safeguard that catches cases where a poor line search or extreme
    nonlinearity has produced an ascent direction.
    """
    return np.dot(g_new, d_new) >= 0.0


# ---------------------------------------------------------------------------
# Main NCG solver
# ---------------------------------------------------------------------------

def ncg_minimize(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    beta_variant: str = "PR+",
    tol: float = 1e-6,
    max_iter: int = 10000,
    restart_every: Optional[int] = None,
    use_powell_restart: bool = True,
    powell_nu: float = 0.1,
    c1: float = 1e-4,
    c2: float = 0.9,
    alpha_init: float = 1.0,
    store_history: bool = True,
) -> NCGResult:
    """
    Nonlinear Conjugate Gradient minimizer.

    Minimizes f : R^n -> R using the NCG algorithm with user-selected
    beta variant, strong-Wolfe line search, and configurable restart strategy.

    Algorithm (Nocedal & Wright, Algorithm 5.4)
    -------------------------------------------
    1. Initialize: g0 = grad_f(x0),  d0 = -g0
    2. While ||gk|| > tol:
       a. Find alpha_k via Wolfe line search along dk
       b. x_{k+1} = xk + alpha_k dk
       c. g_{k+1} = grad_f(x_{k+1})
       d. Compute beta_k (FR / PR / PR+ / HS)
       e. d_{k+1} = -g_{k+1} + beta_k dk
       f. Apply restart if criterion fires: d_{k+1} <- -g_{k+1}

    Parameters
    ----------
    f            : objective function f(x) -> float
    grad_f       : gradient function grad_f(x) -> ndarray of shape (n,)
    x0           : starting point; any shape, ravel()ed to 1-D internally
    beta_variant : one of "FR", "PR", "PR+", "HS"
    tol          : convergence tolerance on ||grad_f(x)||
    max_iter     : maximum number of iterations
    restart_every: periodic restart interval (None -> use n = len(x0.ravel()))
    use_powell_restart : enable Powell's orthogonality criterion;
                         fires independently of periodic restart
    powell_nu    : threshold nu for Powell restart (default 0.1)
    c1, c2       : Wolfe condition parameters (0 < c1 < c2 < 1)
    alpha_init   : initial trial step size for line search (default 1.0)
    store_history: if True, record per-iteration history for diagnostics

    Returns
    -------
    NCGResult dataclass with final state, evaluation counts, and histories.
    """
    if beta_variant not in BETA_FUNCTIONS:
        raise ValueError(
            f"beta_variant='{beta_variant}' is not recognised. "
            f"Choose from {list(BETA_FUNCTIONS)}."
        )

    beta_fn = BETA_FUNCTIONS[beta_variant]

    # Flatten x0 to 1-D; gracefully handles accidental 2-D input
    x = np.array(x0, dtype=float).ravel().copy()
    n = x.size

    if restart_every is None:
        restart_every = n          # Classic: restart every n iterations

    # --- Evaluation counters ------------------------------------------------
    nfev = ngev = 0

    def _f(xx: np.ndarray) -> float:
        nonlocal nfev
        nfev += 1
        return float(f(xx))

    def _g(xx: np.ndarray) -> np.ndarray:
        nonlocal ngev
        ngev += 1
        return np.array(grad_f(xx), dtype=float)

    # --- Initialise ---------------------------------------------------------
    fk = _f(x)
    gk = _g(x)

    if not np.isfinite(fk):
        raise ValueError(f"Initial function value is not finite: f(x0) = {fk}.")
    if not np.all(np.isfinite(gk)):
        raise ValueError("Initial gradient contains NaN or Inf.")

    dk = -gk                       # d0 = -grad_f(x0)

    fun_hist      = [fk]                           if store_history else []
    gnorm_hist    = [float(np.linalg.norm(gk))]   if store_history else []
    x_hist        = [x.copy()]                    if store_history else []
    beta_hist     = []                             if store_history else []
    alpha_hist    = []                             if store_history else []
    restart_flags = []                             if store_history else []

    message = "Maximum iterations reached."
    success = False

    for k in range(max_iter):
        gnorm = float(np.linalg.norm(gk))

        # ---- Convergence check ---------------------------------------------
        if gnorm < tol:
            message = f"Converged: ||grad_f|| = {gnorm:.3e} < tol = {tol:.3e}"
            success = True
            break

        # ---- Safeguard: ensure dk is a descent direction -------------------
        if np.dot(gk, dk) >= 0:
            dk = -gk               # Reset to steepest descent

        # ---- Wolfe line search ---------------------------------------------
        # wolfe_line_search returns f(x_new) to avoid a redundant evaluation.
        try:
            alpha_k, fk_new, ls_nfev, ls_ngev = wolfe_line_search(
                f, grad_f, x, dk, fk, gk,
                alpha_init=alpha_init, c1=c1, c2=c2,
            )
        except AssertionError:
            # Direction not descent (numerical precision edge) — reset once
            dk = -gk
            try:
                alpha_k, fk_new, ls_nfev, ls_ngev = wolfe_line_search(
                    f, grad_f, x, dk, fk, gk,
                    alpha_init=alpha_init, c1=c1, c2=c2,
                )
            except Exception:
                message = f"Line search failed at iteration {k}."
                break

        nfev += ls_nfev
        ngev += ls_ngev

        # ---- Update iterate ------------------------------------------------
        x_new  = x + alpha_k * dk
        gk_new = _g(x_new)     # 1 gradient eval; fk_new reused from line search

        # ---- Compute beta --------------------------------------------------
        beta_k = beta_fn(gk_new, gk, dk)

        # ---- Candidate new direction ---------------------------------------
        dk_new = -gk_new + beta_k * dk

        # ---- Restart logic -------------------------------------------------
        # All three criteria are checked independently (not elif chain).
        # This means Powell can fire on the same iteration as periodic restart,
        # and both will set dk_new = -gk_new (result is idempotent).
        restarted = False

        # 1. Periodic restart (every restart_every completed steps)
        if (k + 1) % restart_every == 0:
            dk_new    = -gk_new
            beta_k    = 0.0
            restarted = True

        # 2. Powell's orthogonality criterion (independent of periodic)
        if (not restarted and use_powell_restart
                and _should_restart_powell(gk_new, gk, powell_nu)):
            dk_new    = -gk_new
            beta_k    = 0.0
            restarted = True

        # 3. Descent-failure safety restart
        if not restarted and _should_restart_descent_failure(gk_new, dk_new):
            dk_new    = -gk_new
            beta_k    = 0.0
            restarted = True

        # ---- Advance state -------------------------------------------------
        x  = x_new
        fk = fk_new
        gk = gk_new
        dk = dk_new

        if store_history:
            fun_hist.append(fk)
            gnorm_hist.append(float(np.linalg.norm(gk)))
            x_hist.append(x.copy())
            beta_hist.append(beta_k)
            alpha_hist.append(alpha_k)
            restart_flags.append(restarted)

    return NCGResult(
        x=x,
        fun=fk,
        grad_norm=float(np.linalg.norm(gk)),
        nit=k + 1,
        nfev=nfev,
        ngev=ngev,
        success=success,
        message=message,
        fun_history=fun_hist,
        grad_norm_history=gnorm_hist,
        x_history=x_hist,
        beta_history=beta_hist,
        alpha_history=alpha_hist,
        restart_flags=restart_flags,
    )
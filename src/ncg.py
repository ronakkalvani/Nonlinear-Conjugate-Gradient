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
    x: np.ndarray
    fun: float
    grad_norm: float
    nit: int
    nfev: int
    ngev: int
    success: bool
    message: str

    # Per-iteration histories (length = completed_steps + 1 for x/fun/grad,
    # length = completed_steps for beta/alpha/restart)
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
    gradient norms. (Fletcher & Reeves, 1964)

    Properties
    ----------
    - Strongest convergence guarantee (global, under strong Wolfe).
    - Can suffer from jamming: a bad step carries forward with full weight
      because FR uses only gradient magnitudes, not directional changes.

    Safeguard: Returns 0.0 when ||g_old|| = 0 to avoid division by zero.
    """
    denom = np.dot(g_old, g_old)
    if denom < 1e-30:
        return 0.0
    return np.dot(g_new, g_new) / denom


def beta_polak_ribiere(g_new: np.ndarray, g_old: np.ndarray,
                        d_old: np.ndarray) -> float:
    """
    Polak-Ribière (PR) beta formula.

        beta_k^PR = g_{k+1}^T (g_{k+1} - g_k) / ||g_k||^2

    Derivation: Uses the gradient difference y_k = g_{k+1} - g_k as a
    curvature proxy. In the quadratic exact case this equals FR (since
    g_{k+1}^T g_k = 0). For nonlinear problems it self-corrects when
    progress stalls: y_k ≈ 0 gives beta ≈ 0, resetting toward steepest
    descent. (Polak & Ribière, 1969)

    Properties
    ----------
    - More robust to bad steps than FR due to self-correcting behaviour.
    - Can be negative, which may violate the descent condition.
    - No global convergence guarantee without truncation (use PR+).

    Safeguard: Returns 0.0 when ||g_old|| = 0 to avoid division by zero.
    """
    denom = np.dot(g_old, g_old)
    if denom < 1e-30:
        return 0.0
    y = g_new - g_old
    return np.dot(g_new, y) / denom


def beta_polak_ribiere_plus(g_new: np.ndarray, g_old: np.ndarray,
                             d_old: np.ndarray) -> float:
    """
    Polak-Ribière+ (PR+) beta formula — Powell (1984).

        beta_k^PR+ = max(beta_k^PR, 0)

    Rationale: Truncating negative PR values at zero prevents the direction
    from violating the descent condition. When beta = 0, dk+1 = -g_{k+1},
    which is steepest descent — an automatic restart. This gives PR+ global
    convergence under mild conditions while retaining PR's empirical
    efficiency. Best overall for general nonlinear optimization.
    """
    return max(beta_polak_ribiere(g_new, g_old, d_old), 0.0)


def beta_hestenes_stiefel(g_new: np.ndarray, g_old: np.ndarray,
                           d_old: np.ndarray) -> float:
    """
    Hestenes-Stiefel (HS) beta formula.

        beta_k^HS = g_{k+1}^T (g_{k+1} - g_k) / d_k^T (g_{k+1} - g_k)

    Derivation: Directly imposes the secant-based conjugacy condition
    d_{k+1}^T y_k = 0, where y_k = g_{k+1} - g_k. Substituting
    d_{k+1} = -g_{k+1} + beta_k d_k and solving for beta_k yields this
    formula. This is the most conjugacy-motivated variant and is consistent
    with the secant equation used in quasi-Newton methods.

    Properties
    ----------
    - Theoretically well-motivated; directly enforces d_{k+1} perp y_k.
    - Competitive when line search quality is high.
    - No global convergence guarantee without additional safeguards.

    Safeguard: Returns 0.0 when |d_k^T y_k| < 1e-30 to avoid near-zero
    denominator. This triggers an implicit steepest-descent restart.
    """
    y = g_new - g_old
    denom = np.dot(d_old, y)
    if abs(denom) < 1e-30:
        return 0.0
    return np.dot(g_new, y) / denom


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
    Strong Wolfe line search with bracketing and zoom phases.

    Finds alpha_k satisfying the strong Wolfe conditions:
        (1) f(x + alpha d) <= f(x) + c1 alpha g0^T d    (Armijo)
        (2) |grad_f(x + alpha d)^T d| <= c2 |g0^T d|    (strong curvature)

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
    c2         : strong curvature constant (typically 0.9 for NCG)
    alpha_max  : maximum allowed step size
    max_iter   : iteration budget for bracketing and zoom phases

    Returns
    -------
    alpha  : accepted step size
    f_new  : f(x + alpha * d) — cached to avoid a redundant call in the caller
    nfev   : number of function evaluations used
    ngev   : number of gradient evaluations used
    """
    import warnings

    dphi0 = np.dot(g0, d)
    assert dphi0 < 0, (
        f"Search direction is not a descent direction: g^T d = {dphi0:.3e} >= 0."
    )

    nfev = ngev = 0

    def phi(a: float) -> float:
        nonlocal nfev; nfev += 1; return f(x + a * d)

    def dphi(a: float) -> float:
        nonlocal ngev; ngev += 1; return np.dot(grad_f(x + a * d), d)

    def phi_dphi(a: float) -> Tuple[float, float]:
        nonlocal nfev, ngev; nfev += 1; ngev += 1
        xa = x + a * d; return f(xa), np.dot(grad_f(xa), d)

    def zoom(alpha_lo: float, alpha_hi: float,
             phi_lo: float, phi_hi: float,
             dphi_lo: float, dphi_hi_init: float) -> Tuple[float, float]:
        """
        Narrow [alpha_lo, alpha_hi] until strong Wolfe conditions hold.

        Invariants:
          - phi(alpha_lo) satisfies Armijo.
          - phi(alpha_lo) <= phi(alpha_hi).
          - The minimiser of phi lies within the interval.
        """
        dphi_hi = dphi_hi_init  # cache; update only when alpha_hi changes
        for _ in range(max_iter):
            alpha_j = _cubic_interp(alpha_lo, alpha_hi, phi_lo, phi_hi, dphi_lo, dphi_hi)
            lo = min(alpha_lo, alpha_hi); hi = max(alpha_lo, alpha_hi)
            if alpha_j <= lo or alpha_j >= hi:
                alpha_j = 0.5 * (alpha_lo + alpha_hi)
            phi_j, dphi_j = phi_dphi(alpha_j)
            if (phi_j > f0 + c1 * alpha_j * dphi0) or (phi_j >= phi_lo):
                alpha_hi = alpha_j; phi_hi = phi_j; dphi_hi = dphi_j
            else:
                if abs(dphi_j) <= c2 * abs(dphi0):
                    return alpha_j, phi_j
                if dphi_j * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo; phi_hi = phi_lo; dphi_hi = dphi_lo
                alpha_lo = alpha_j; phi_lo = phi_j; dphi_lo = dphi_j
        warnings.warn(
            "Wolfe zoom exhausted iteration budget without satisfying the "
            "strong curvature condition. Returning best Armijo-satisfying step.",
            RuntimeWarning, stacklevel=4)
        return alpha_lo, phi_lo

    # Bracketing phase
    alpha_prev = 0.0; phi_prev = f0; dphi_prev = dphi0
    alpha_cur  = alpha_init

    for i in range(max_iter):
        phi_cur = phi(alpha_cur)
        if (phi_cur > f0 + c1 * alpha_cur * dphi0) or (i > 0 and phi_cur >= phi_prev):
            alpha_star, f_star = zoom(
                alpha_prev, alpha_cur, phi_prev, phi_cur, dphi_prev, dphi(alpha_cur))
            return alpha_star, f_star, nfev, ngev
        dphi_cur = dphi(alpha_cur)
        if abs(dphi_cur) <= c2 * abs(dphi0):
            return alpha_cur, phi_cur, nfev, ngev
        if dphi_cur >= 0:
            alpha_star, f_star = zoom(
                alpha_cur, alpha_prev, phi_cur, phi_prev, dphi_cur, dphi_prev)
            return alpha_star, f_star, nfev, ngev
        alpha_prev = alpha_cur; phi_prev = phi_cur; dphi_prev = dphi_cur
        alpha_cur  = min(2.0 * alpha_cur, alpha_max)

    f_cur = phi(alpha_cur)
    warnings.warn(
        "Wolfe bracketing exhausted iteration budget. "
        "Returning last step without Wolfe guarantee.",
        RuntimeWarning, stacklevel=3)
    return alpha_cur, f_cur, nfev, ngev


def _cubic_interp(a1: float, a2: float,
                  f1: float, f2: float,
                  df1: float, df2: float) -> float:
    """Cubic Hermite interpolation; bisection fallback when degenerate."""
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
    return float(a2 - (a2 - a1) * (df2 + d2 - d1) / denom)


# ---------------------------------------------------------------------------
# Restart criteria
# ---------------------------------------------------------------------------

def _should_restart_powell(g_new: np.ndarray, g_old: np.ndarray,
                            nu: float = 0.1) -> bool:
    """
    Powell's restart criterion (Powell 1977).

    Restart when consecutive gradients are insufficiently orthogonal:
        |g_{k+1}^T g_k| / ||g_{k+1}||^2 >= nu

    A large ratio indicates near-parallel gradients (lost conjugacy).
    Typical threshold: nu = 0.1.
    """
    denom = np.dot(g_new, g_new)
    if denom < 1e-30:
        return False
    return abs(np.dot(g_new, g_old)) / denom >= nu


def _should_restart_descent_failure(g_new: np.ndarray,
                                    d_new: np.ndarray) -> bool:
    """
    Descent-failure restart: trigger when g_{k+1}^T d_{k+1} >= 0.

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

    Minimizes f : R^n -> R using the NCG algorithm with the selected beta
    variant, strong-Wolfe line search, and configurable restart strategy.

    Algorithm (Nocedal & Wright, Algorithm 5.4 / PDF Section 3.1)
    --------------------------------------------------------------
    1. Initialize: g0 = grad_f(x0),  d0 = -g0
    2. While ||gk|| > tol:
       a. Find alpha_k via Wolfe line search along dk
       b. x_{k+1} = xk + alpha_k dk
       c. g_{k+1} = grad_f(x_{k+1})
       d. Compute beta_k (FR / PR / PR+ / HS)
       e. d_{k+1} = -g_{k+1} + beta_k dk
       f. Apply restart if criterion fires: d_{k+1} <- -g_{k+1}
    3. Return xk

    Memory: O(n) — stores only xk, gk, g_{k-1}, dk.

    Parameters
    ----------
    f            : objective function f(x) -> float
    grad_f       : gradient grad_f(x) -> ndarray shape (n,)
    x0           : starting point; any shape, ravel()ed to 1-D internally
    beta_variant : one of "FR", "PR", "PR+", "HS"
    tol          : convergence tolerance on ||grad_f(x)||
    max_iter     : maximum number of iterations
    restart_every: periodic restart interval (None -> n = len(x))
    use_powell_restart : enable Powell's orthogonality criterion
    powell_nu    : threshold nu for Powell restart (default 0.1)
    c1, c2       : Wolfe parameters (0 < c1 < c2 < 1), default 1e-4 / 0.9
    alpha_init   : initial trial step size for line search (default 1.0)
    store_history: record per-iteration diagnostics

    Returns
    -------
    NCGResult dataclass with final state, evaluation counts, and histories.
    """
    if beta_variant not in BETA_FUNCTIONS:
        raise ValueError(
            f"beta_variant='{beta_variant}' is not recognised. "
            f"Choose from {list(BETA_FUNCTIONS)}.")

    beta_fn = BETA_FUNCTIONS[beta_variant]
    x = np.array(x0, dtype=float).ravel().copy()
    n = x.size

    if restart_every is None:
        restart_every = n

    nfev = ngev = 0

    def _f(xx: np.ndarray) -> float:
        nonlocal nfev; nfev += 1; return float(f(xx))

    def _g(xx: np.ndarray) -> np.ndarray:
        nonlocal ngev; ngev += 1; return np.array(grad_f(xx), dtype=float)

    fk = _f(x); gk = _g(x)

    if not np.isfinite(fk):
        raise ValueError(f"Initial function value is not finite: f(x0) = {fk}.")
    if not np.all(np.isfinite(gk)):
        raise ValueError("Initial gradient contains NaN or Inf.")

    dk = -gk

    fun_hist      = [fk]                          if store_history else []
    gnorm_hist    = [float(np.linalg.norm(gk))]   if store_history else []
    x_hist        = [x.copy()]                    if store_history else []
    beta_hist     = []
    alpha_hist    = []
    restart_flags = []

    message = "Maximum iterations reached."
    success = False

    for k in range(max_iter):
        gnorm = float(np.linalg.norm(gk))

        # Convergence check (Section 5: ||nabla f(xk)|| -> 0)
        if gnorm < tol:
            message = f"Converged: ||grad_f|| = {gnorm:.3e} < tol = {tol:.3e}"
            success = True
            break

        # Descent-direction safeguard
        if np.dot(gk, dk) >= 0:
            dk = -gk

        # Wolfe line search (returns f_new to avoid redundant evaluation)
        try:
            alpha_k, fk_new, ls_nfev, ls_ngev = wolfe_line_search(
                f, grad_f, x, dk, fk, gk,
                alpha_init=alpha_init, c1=c1, c2=c2)
        except AssertionError:
            dk = -gk
            try:
                alpha_k, fk_new, ls_nfev, ls_ngev = wolfe_line_search(
                    f, grad_f, x, dk, fk, gk,
                    alpha_init=alpha_init, c1=c1, c2=c2)
            except Exception:
                message = f"Line search failed at iteration {k}."; break

        nfev += ls_nfev; ngev += ls_ngev

        # Update iterate and gradient
        x_new  = x + alpha_k * dk
        gk_new = _g(x_new)

        # Compute beta and new direction
        beta_k = beta_fn(gk_new, gk, dk)
        dk_new = -gk_new + beta_k * dk

        # Restart logic — all three criteria fire independently (not elif chain)
        # This ensures Powell can fire on the same iteration as periodic restart.
        restarted = False

        # 1. Periodic restart (every restart_every completed steps)
        if (k + 1) % restart_every == 0:
            dk_new = -gk_new; beta_k = 0.0; restarted = True

        # 2. Powell's orthogonality criterion (independent of periodic)
        if (not restarted and use_powell_restart
                and _should_restart_powell(gk_new, gk, powell_nu)):
            dk_new = -gk_new; beta_k = 0.0; restarted = True

        # 3. Descent-failure safety restart
        if not restarted and _should_restart_descent_failure(gk_new, dk_new):
            dk_new = -gk_new; beta_k = 0.0; restarted = True

        # Advance state
        x = x_new; fk = fk_new; gk = gk_new; dk = dk_new

        if store_history:
            fun_hist.append(fk)
            gnorm_hist.append(float(np.linalg.norm(gk)))
            x_hist.append(x.copy())
            beta_hist.append(beta_k)
            alpha_hist.append(alpha_k)
            restart_flags.append(restarted)

    return NCGResult(
        x=x, fun=fk, grad_norm=float(np.linalg.norm(gk)),
        nit=k + 1, nfev=nfev, ngev=ngev,
        success=success, message=message,
        fun_history=fun_hist, grad_norm_history=gnorm_hist, x_history=x_hist,
        beta_history=beta_hist, alpha_history=alpha_hist, restart_flags=restart_flags,
    )
# Nonlinear Conjugate Gradient (NCG) — Complete Implementation

**MTL704 Assignment - IIT Delhi, March 2026**

*Dept. of Mathematics and Computing | Prof. Aparna Mehra*

---

## Project Structure

```
ncg/
├── main.py                  # Master runner (tests + benchmarks + plots)
├── src/
│   ├── ncg.py               # Core NCG algorithm
│   ├── test_functions.py    # Benchmark test functions
│   ├── comparison.py        # Gradient Descent & scipy comparison runner
|   ├── run_tests.py         # Standalone test runner
│   └── plots.py             # All convergence demonstration plots
├── tests/
│   └── test_ncg.py          # Full pytest unit-test suite
└── results/                 # Generated plots and results
```

---

## Quick Start

```bash
# Install dependencies (only numpy, scipy, matplotlib needed)
pip install numpy scipy matplotlib pytest

# Run the quick demo (Rosenbrock, all 4 variants)
python main.py --demo

# Run unit tests only
python main.py --tests-only

# Generate all plots only (skips tests)
python main.py --plots-only

# Run full pipeline: tests + benchmarks + plots
python main.py
```

---

## Module Reference

### `src/ncg.py` — Core Algorithm

#### `ncg_minimize(f, grad_f, x0, beta_variant="PR+", ...)`
Main solver. Returns an `NCGResult` dataclass with:
- `x`, `fun`, `grad_norm` — final state
- `nit`, `nfev`, `ngev` — iteration and evaluation counts
- `fun_history`, `grad_norm_history`, `x_history` — per-iteration logs
- `beta_history`, `alpha_history`, `restart_flags` — diagnostic logs

**Beta variants:** `"FR"`, `"PR"`, `"PR+"`, `"HS"`

```python
from src.ncg import ncg_minimize
import numpy as np

f    = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
gf   = lambda x: np.array([
    -2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2),
     200*(x[1] - x[0]**2)
])
res = ncg_minimize(f, gf, np.array([-1.2, 1.0]), beta_variant="PR+")
print(res.x)   # → [1.0, 1.0]
```

#### Beta functions (all accept `g_new, g_old, d_old`):
| Function | Formula |
|---|---|
| `beta_fletcher_reeves` | ‖g_{k+1}‖² / ‖g_k‖² |
| `beta_polak_ribiere` | g_{k+1}^T y_k / ‖g_k‖² |
| `beta_polak_ribiere_plus` | max(β^PR, 0) |
| `beta_hestenes_stiefel` | g_{k+1}^T y_k / d_k^T y_k |

#### `wolfe_line_search(f, grad_f, x, d, f0, g0, ...)`
Bracketing + zoom Wolfe line search. Enforces strong Wolfe conditions.

---

### `src/test_functions.py` — Benchmark Problems

| Function | Dim | Minimizer | Notes |
|---|---|---|---|
| `quadratic_bowl_2d(kappa)` | 2 | (0,0) | Condition number κ |
| `rosenbrock()` | 2 | (1,1) | Classic banana valley |
| `beale()` | 2 | (3, 0.5) | Multiple local structures |
| `high_dim_quadratic(n, kappa)` | n | A⁻¹b | Tests finite-step property |

All return `(f, grad_f, x_star, f_star)`.

---

### `src/comparison.py` — Method Comparison

`run_all_benchmarks(tol)` — runs all NCG variants + GD + scipy on all problems,
returns a list of `BenchmarkRow` dataclasses.

---

### `src/plots.py` — Convergence Plots

All figures saved to `results/`:

| Filename | Description |
|---|---|
| `loss_vs_iter_*.png` | f(xk) − f* vs iterations (log scale) |
| `grad_norm_*.png` | ‖∇f(xk)‖ vs iterations |
| `error_vs_iter_*.png` | ‖xk − x*‖ vs iterations |
| `trajectory_*.png` | 2-D path on contour plot |
| `beta_alpha_*.png` | βk and αk diagnostic plots |
| `dashboard_*.png` | 4-panel convergence summary |
| `comparison_bar.png` | Iterations to convergence (all methods) |
| `finite_termination.png` | Iterations vs κ (quadratic bound study) |

---

## Algorithm Summary

```
Algorithm: NCG (Nonlinear Conjugate Gradient)
─────────────────────────────────────────────
Input: f, ∇f, x₀, β-variant, tol, max_iter

1.  g₀ ← ∇f(x₀),  d₀ ← −g₀
2.  for k = 0, 1, 2, ...:
3.      if ‖gk‖ < tol: STOP  (converged)
4.      αk ← Wolfe_line_search(f, ∇f, xk, dk)
5.      x_{k+1} ← xk + αk dk
6.      g_{k+1} ← ∇f(x_{k+1})
7.      βk ← beta_variant(g_{k+1}, gk, dk)
8.      d_{k+1} ← −g_{k+1} + βk dk
9.      if restart triggered: d_{k+1} ← −g_{k+1}
10. return xk
```

**Memory:** O(n) — stores only xk, gk, g_{k-1}, dk.
**Per-iteration cost:** O(n) gradient evaluation + O(n) scalar ops.

---

## Beta Variant Comparison

| Variant | Global Conv. | Descent Guar. | Practical Perf. |
|---|---|---|---|
| Fletcher-Reeves (FR) | ✓ (strong Wolfe) | Conditional | Can jam |
| Polak-Ribière (PR) | ✗ | ✗ | Good, robust |
| Polak-Ribière+ (PR+) | ✓ (mild cond.) | ✓ | **Best overall** |
| Hestenes-Stiefel (HS) | ✗ | ✗ | Competitive |

**Recommendation:** Use `PR+` by default.

---

## References

- Fletcher & Reeves (1964). *Function minimization by conjugate gradients.* Comput. J.
- Polak & Ribière (1969). *Note sur la convergence de méthodes de directions conjuguées.*
- Nocedal & Wright (2006). *Numerical Optimization*, 2nd ed. Springer.
- Hager & Zhang (2006). *A survey of nonlinear conjugate gradient methods.*
- Shewchuk (1994). *An Introduction to the CG Method Without the Agonizing Pain.*

"""
main.py — NCG Master Runner
============================
Executes the full pipeline:
  1. Unit tests (via pytest)
  2. Benchmark comparisons across all problems & methods
  3. All convergence demonstration plots

Run from the project root:
    python main.py
    python main.py --plots-only
    python main.py --tests-only
    python main.py --benchmark-only
"""

import sys, os, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np


def run_tests():
    """Execute pytest unit tests."""
    import subprocess
    print("\n" + "="*60)
    print("  UNIT TESTS")
    print("="*60)
    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         os.path.join(os.path.dirname(__file__), "tests"),
         "-v", "--tb=short"],
        capture_output=False
    )
    return result.returncode == 0


def run_benchmarks():
    """Run method comparison on all test problems."""
    from comparison import run_all_benchmarks, print_summary_table
    rows = run_all_benchmarks(tol=1e-7)
    print_summary_table(rows)
    return rows


def run_plots():
    """Generate all convergence plots."""
    from plots import generate_all_plots
    generate_all_plots()


def quick_demo():
    """
    Short self-contained demo: runs NCG-PR+ on Rosenbrock and prints progress.
    Useful for a quick sanity-check without the full pipeline.
    """
    from ncg import ncg_minimize
    from test_functions import rosenbrock

    print("\n" + "="*60)
    print("  QUICK DEMO: NCG-PR+ on Rosenbrock  f(x,y) = (1-x)² + 100(y-x²)²")
    print("  Starting at x0 = (-1.2, 1.0)  →  Minimum at x* = (1, 1)")
    print("="*60)

    f, gf, xs, fs = rosenbrock()
    x0 = np.array([-1.2, 1.0])

    # Run all four variants and compare
    from ncg import BETA_FUNCTIONS
    print(f"\n{'Variant':<8} {'Iter':>6}  {'f(x*)':>12}  {'‖∇f‖':>12}  {'‖x-x*‖':>12}  {'Status'}")
    print("-"*70)
    for variant in ["FR", "PR", "PR+", "HS"]:
        res = ncg_minimize(f, gf, x0.copy(), beta_variant=variant,
                           tol=1e-8, max_iter=5000)
        xerr = np.linalg.norm(res.x - xs)
        status = "✓ converged" if res.success else "✗ max_iter"
        print(f"{variant:<8} {res.nit:>6}  {res.fun:>12.6e}  "
              f"{res.grad_norm:>12.6e}  {xerr:>12.6e}  {status}")

    print("\nNote: PR+ is the recommended default for general nonlinear optimization.")
    print("      FR provides the strongest theoretical guarantee (global convergence")
    print("      under strong Wolfe) but may jam on ill-conditioned problems.")


def main():
    parser = argparse.ArgumentParser(description="NCG benchmark & plot runner")
    parser.add_argument("--tests-only",     action="store_true")
    parser.add_argument("--plots-only",     action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--demo",           action="store_true",
                        help="Run quick demo only")
    args = parser.parse_args()

    t_start = time.perf_counter()

    if args.demo:
        quick_demo()
    elif args.tests_only:
        run_tests()
    elif args.plots_only:
        run_plots()
    elif args.benchmark_only:
        run_benchmarks()
    else:
        # Full pipeline
        ok = run_tests()
        if not ok:
            print("\nWARNING: Some tests failed. Continuing with benchmarks & plots.")
        run_benchmarks()
        run_plots()
        quick_demo()

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal wall time: {elapsed:.1f} s")


if __name__ == "__main__":
    main()

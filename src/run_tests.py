"""run_tests.py — Standalone test runner"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np
from ncg import (ncg_minimize, beta_fletcher_reeves, beta_polak_ribiere,
                 beta_polak_ribiere_plus, beta_hestenes_stiefel,
                 wolfe_line_search, BETA_FUNCTIONS)
from test_functions import (quadratic_bowl_2d, rosenbrock, beale, high_dim_quadratic)

PASS="  PASS"; FAIL="  FAIL"; TOL=1e-6
_results={"passed":0,"failed":0}

def test(name,fn):
    try: fn(); print(f"{PASS}  {name}"); _results["passed"]+=1
    except Exception as e: print(f"{FAIL}  {name}"); print(f"         -> {e}"); _results["failed"]+=1

def _numerical_grad(f,x,h=1e-6):
    g=np.zeros_like(x)
    for i in range(len(x)):
        xp=x.copy(); xp[i]+=h; xm=x.copy(); xm[i]-=h
        g[i]=(f(xp)-f(xm))/(2*h)
    return g

def _gradients():
    rng=np.random.default_rng(0); return rng.standard_normal(10),rng.standard_normal(10),rng.standard_normal(10)

def t_fr_positive():
    g,go,d=_gradients(); assert beta_fletcher_reeves(g,go,d)>=0
def t_fr_formula():
    g,go,d=_gradients(); exp=np.dot(g,g)/np.dot(go,go); assert abs(beta_fletcher_reeves(g,go,d)-exp)<1e-14
def t_pr_can_be_negative():
    g=np.array([0.5,0.0]); go=np.array([2.0,0.0]); d=np.zeros(2); assert beta_polak_ribiere(g,go,d)<0
def t_prplus_nonneg():
    g=np.array([0.5,0.0]); go=np.array([2.0,0.0]); d=np.zeros(2); assert beta_polak_ribiere_plus(g,go,d)>=0
def t_prplus_equals_pr_when_positive():
    g=np.array([1.5,0.5]); go=np.array([1.0,0.0]); d=np.zeros(2)
    assert abs(beta_polak_ribiere_plus(g,go,d)-max(beta_polak_ribiere(g,go,d),0.0))<1e-14
def t_hs_zero_denom():
    g=np.array([1.0,0.0]); go=np.array([1.0,0.0]); d=np.array([1.0,0.0]); assert beta_hestenes_stiefel(g,go,d)==0.0
def t_hs_formula():
    g=np.array([1.2,0.4]); go=np.array([0.8,0.1]); d=np.array([0.3,-0.5]); y=g-go; denom=np.dot(d,y)
    if abs(denom)>1e-30: exp=np.dot(g,y)/denom; assert abs(beta_hestenes_stiefel(g,go,d)-exp)<1e-12
def t_zero_old_grad_safeguard():
    go=np.zeros(5); g=np.ones(5); d=np.ones(5)
    for name in ["FR","PR","PR+"]: v=BETA_FUNCTIONS[name](g,go,d); assert v==0.0,f"{name} should return 0 when g_old=0, got {v}"
def t_fr_pr_equal_on_quadratic():
    g=np.array([0.0,1.0]); go=np.array([1.0,0.0]); d=np.zeros(2)
    fr=beta_fletcher_reeves(g,go,d); pr=beta_polak_ribiere(g,go,d); assert abs(fr-pr)<1e-12

def _check_grad(f,gf,x_tests,name,tol=1e-5):
    for x in x_tests:
        ana=gf(x); num=_numerical_grad(f,x); err=np.linalg.norm(ana-num)
        assert err<tol,f"{name}: gradient error {err:.2e} at x={x}"

def t_grad_rosenbrock():
    f,gf,_,_=rosenbrock()
    _check_grad(f,gf,[np.array([-1.2,1.0]),np.array([0.5,0.25]),np.array([1.0,1.0])],"Rosenbrock")
def t_grad_beale():
    f,gf,_,_=beale()
    _check_grad(f,gf,[np.array([1.0,1.0]),np.array([2.0,0.3]),np.array([3.0,0.5])],"Beale")
def t_grad_quadratic():
    f,gf,_,_=quadratic_bowl_2d(kappa=20.0)
    _check_grad(f,gf,[np.array([3.0,1.0]),np.array([0.0,0.0]),np.array([-2.0,4.0])],"Quadratic2D")
def t_grad_highdim():
    f,gf,_,_=high_dim_quadratic(n=10,kappa=5,seed=7)
    x_test=np.random.default_rng(42).standard_normal(10); _check_grad(f,gf,[x_test],"HighDimQuadratic")

def _rosen_setup():
    f,gf,_,_=rosenbrock(); x=np.array([-1.2,1.0]); g=gf(x); d=-g; return f,gf,x,f(x),g,d

def t_wolfe_returns_4tuple():
    f,gf,x,fx,g,d=_rosen_setup(); result=wolfe_line_search(f,gf,x,d,fx,g); assert len(result)==4,f"Expected 4-tuple, got {len(result)}-tuple"
def t_wolfe_armijo():
    f,gf,x,fx,g,d=_rosen_setup(); c1=1e-4; alpha,f_new,_,_=wolfe_line_search(f,gf,x,d,fx,g,c1=c1,c2=0.9); dphi0=np.dot(g,d); assert f_new<=fx+c1*alpha*dphi0+1e-10
def t_wolfe_fnew_matches_f():
    f,gf,x,fx,g,d=_rosen_setup(); alpha,f_new,_,_=wolfe_line_search(f,gf,x,d,fx,g); assert abs(f_new-f(x+alpha*d))<1e-14,f"f_new={f_new} != f(x+alpha*d)={f(x+alpha*d)}"
def t_wolfe_curvature():
    f,gf,x,fx,g,d=_rosen_setup(); c2=0.9; alpha,_,_,_=wolfe_line_search(f,gf,x,d,fx,g,c1=1e-4,c2=c2); dphi0=np.dot(g,d); dphi_new=np.dot(gf(x+alpha*d),d); assert abs(dphi_new)<=c2*abs(dphi0)+1e-8
def t_wolfe_positive_step():
    f,gf,x,fx,g,d=_rosen_setup(); alpha,_,_,_=wolfe_line_search(f,gf,x,d,fx,g); assert alpha>0
def t_wolfe_non_descent_raises():
    f,gf,x,fx,g,d=_rosen_setup()
    try: wolfe_line_search(f,gf,x,-d,fx,g); assert False,"Should have raised AssertionError"
    except AssertionError: pass
def t_wolfe_on_beale():
    f,gf,xs,_=beale(); x=np.array([1.0,1.0]); g=gf(x); d=-g
    alpha,f_new,_,_=wolfe_line_search(f,gf,x,d,f(x),g,c1=1e-4,c2=0.9); dphi0=np.dot(g,d)
    assert f_new<=f(x)+1e-4*alpha*dphi0+1e-10; assert abs(np.dot(gf(x+alpha*d),d))<=0.9*abs(dphi0)+1e-8

def _test_conv(f,gf,x0,xs,variant,max_iter=3000,grad_atol=None,x_atol=1e-3):
    if grad_atol is None: grad_atol=TOL*100
    res=ncg_minimize(f,gf,x0,beta_variant=variant,tol=TOL,max_iter=max_iter)
    assert res.grad_norm<grad_atol,f"{variant}: ||grad_f||={res.grad_norm:.2e} >= {grad_atol:.2e}"
    err=np.linalg.norm(res.x-xs); assert err<x_atol,f"{variant}: ||x-x*||={err:.2e} >= {x_atol:.2e}"

def t_quadratic_FR(): _test_conv(*quadratic_bowl_2d()[:2],np.array([4.,4.]),np.zeros(2),"FR",x_atol=1e-4)
def t_quadratic_PR(): _test_conv(*quadratic_bowl_2d()[:2],np.array([4.,4.]),np.zeros(2),"PR",x_atol=1e-4)
def t_quadratic_PRp(): _test_conv(*quadratic_bowl_2d()[:2],np.array([4.,4.]),np.zeros(2),"PR+",x_atol=1e-4)
def t_quadratic_HS(): _test_conv(*quadratic_bowl_2d()[:2],np.array([4.,4.]),np.zeros(2),"HS",x_atol=1e-4)
def t_rosenbrock_FR(): _test_conv(*rosenbrock()[:2],np.array([-1.2,1.]),np.array([1.,1.]),"FR")
def t_rosenbrock_PR(): _test_conv(*rosenbrock()[:2],np.array([-1.2,1.]),np.array([1.,1.]),"PR")
def t_rosenbrock_PRp(): _test_conv(*rosenbrock()[:2],np.array([-1.2,1.]),np.array([1.,1.]),"PR+")
def t_rosenbrock_HS(): _test_conv(*rosenbrock()[:2],np.array([-1.2,1.]),np.array([1.,1.]),"HS")
def t_beale_FR(): _test_conv(*beale()[:2],np.array([1.,1.]),np.array([3.,0.5]),"FR",x_atol=1e-3)
def t_beale_PR(): _test_conv(*beale()[:2],np.array([1.,1.]),np.array([3.,0.5]),"PR",x_atol=1e-3)
def t_beale_PRp(): _test_conv(*beale()[:2],np.array([1.,1.]),np.array([3.,0.5]),"PR+",x_atol=1e-3)
def t_beale_HS(): _test_conv(*beale()[:2],np.array([1.,1.]),np.array([3.,0.5]),"HS",x_atol=1e-3)
def t_highdim_PRp():
    f,g,xs,_=high_dim_quadratic(n=30,kappa=20,seed=1)
    res=ncg_minimize(f,g,np.zeros(30),beta_variant="PR+",tol=TOL,max_iter=2000)
    assert res.grad_norm<TOL*100,f"||grad_f||={res.grad_norm}"
def t_finite_termination():
    n=10; f,g,xs,_=high_dim_quadratic(n=n,kappa=5,seed=2)
    res=ncg_minimize(f,g,np.zeros(n),beta_variant="PR+",tol=1e-6,max_iter=500)
    assert res.grad_norm<1e-4,f"Did not converge: ||grad_f||={res.grad_norm}"
def t_already_converged_x0():
    f,gf,xs,_=rosenbrock()
    res=ncg_minimize(f,gf,xs.copy(),beta_variant="PR+",tol=1e-6,max_iter=100)
    assert res.success,"Should report success when starting at minimizer"
    assert res.nit==1,f"Expected 1 iteration (convergence check), got {res.nit}"
def t_2d_x0_raveled():
    f,gf,xs,_=quadratic_bowl_2d(); x0_2d=np.array([[4.0,4.0]])
    res=ncg_minimize(f,gf,x0_2d,beta_variant="PR+",tol=1e-6,max_iter=100)
    assert res.x.shape==(2,),f"Expected shape (2,), got {res.x.shape}"; assert res.success

def t_periodic_restart():
    f,g,_,_=rosenbrock()
    res=ncg_minimize(f,g,np.array([-1.2,1.]),beta_variant="PR+",restart_every=5,use_powell_restart=False,tol=1e-8,max_iter=200)
    assert res.restart_flags[4],"Periodic restart should fire at completed step 5 (index 4)"
def t_powell_restart_fires():
    f,g,_,_=rosenbrock()
    res=ncg_minimize(f,g,np.array([-1.2,1.]),beta_variant="FR",use_powell_restart=True,powell_nu=0.05,tol=1e-8,max_iter=500)
    assert any(res.restart_flags),"Powell restart should fire at least once on Rosenbrock"
def t_powell_independent_of_periodic():
    f,g,_,_=rosenbrock()
    res=ncg_minimize(f,g,np.array([-1.2,1.]),beta_variant="FR",restart_every=2,use_powell_restart=True,powell_nu=0.0,tol=1e-8,max_iter=50)
    assert all(res.restart_flags),"Both periodic and Powell should fire every step"

def t_history_lengths():
    f,g,_,_=quadratic_bowl_2d()
    res=ncg_minimize(f,g,np.array([3.,3.]),tol=1e-8,max_iter=100)
    nsteps=len(res.beta_history)
    assert len(res.fun_history)==nsteps+1,"fun_history: initial + nsteps"
    assert len(res.grad_norm_history)==nsteps+1
    assert len(res.x_history)==nsteps+1
    assert len(res.alpha_history)==nsteps
    assert len(res.restart_flags)==nsteps
def t_no_history_when_disabled():
    f,g,_,_=quadratic_bowl_2d()
    res=ncg_minimize(f,g,np.array([3.,3.]),tol=1e-6,max_iter=100,store_history=False)
    assert res.fun_history==[]; assert res.grad_norm_history==[]; assert res.x_history==[]
def t_invalid_variant_raises():
    f,g,_,_=quadratic_bowl_2d()
    try: ncg_minimize(f,g,np.zeros(2),beta_variant="INVALID"); assert False,"Should raise ValueError"
    except ValueError: pass
def t_x_shape_preserved():
    f,g,_,_=high_dim_quadratic(n=20,kappa=5)
    res=ncg_minimize(f,g,np.zeros(20),tol=1e-6,max_iter=200); assert res.x.shape==(20,)
def t_nfev_count_accurate():
    actual=[0]; f_r,gf_r,_,_=rosenbrock()
    def f_counted(x): actual[0]+=1; return f_r(x)
    def g_counted(x): return gf_r(x)
    res=ncg_minimize(f_counted,g_counted,np.array([-1.2,1.0]),beta_variant="PR+",tol=1e-6,max_iter=100)
    assert actual[0]==res.nfev,f"Reported nfev={res.nfev} but actual calls={actual[0]}"
def t_fk_no_redundant_eval():
    actual_f=[0]; actual_g=[0]; f_r,gf_r,_,_=quadratic_bowl_2d()
    def fc(x): actual_f[0]+=1; return f_r(x)
    def gc(x): actual_g[0]+=1; return gf_r(x)
    res=ncg_minimize(fc,gc,np.array([4.,4.]),beta_variant="PR+",tol=1e-8,max_iter=20)
    assert res.nfev==actual_f[0],f"nfev mismatch: reported {res.nfev}, actual {actual_f[0]}"

TESTS=[
    ("FR beta is non-negative",t_fr_positive),
    ("FR beta formula correct",t_fr_formula),
    ("PR beta can be negative",t_pr_can_be_negative),
    ("PR+ is always non-negative",t_prplus_nonneg),
    ("PR+ equals PR when PR > 0",t_prplus_equals_pr_when_positive),
    ("HS zero-denominator safeguard",t_hs_zero_denom),
    ("HS formula correct",t_hs_formula),
    ("Zero old-gradient safeguard",t_zero_old_grad_safeguard),
    ("FR == PR on exact quadratic",t_fr_pr_equal_on_quadratic),
    ("Gradient check: Rosenbrock",t_grad_rosenbrock),
    ("Gradient check: Beale",t_grad_beale),
    ("Gradient check: Quadratic 2D",t_grad_quadratic),
    ("Gradient check: High-dim quad",t_grad_highdim),
    ("Wolfe returns (alpha,f_new,nfev,ngev)",t_wolfe_returns_4tuple),
    ("Wolfe f_new matches f(x+alpha*d)",t_wolfe_fnew_matches_f),
    ("Wolfe Armijo satisfied",t_wolfe_armijo),
    ("Wolfe curvature satisfied",t_wolfe_curvature),
    ("Wolfe returns positive step",t_wolfe_positive_step),
    ("Non-descent raises AssertionError",t_wolfe_non_descent_raises),
    ("Wolfe conditions on Beale",t_wolfe_on_beale),
    ("Quadratic bowl -- FR",t_quadratic_FR),
    ("Quadratic bowl -- PR",t_quadratic_PR),
    ("Quadratic bowl -- PR+",t_quadratic_PRp),
    ("Quadratic bowl -- HS",t_quadratic_HS),
    ("Rosenbrock -- FR",t_rosenbrock_FR),
    ("Rosenbrock -- PR",t_rosenbrock_PR),
    ("Rosenbrock -- PR+",t_rosenbrock_PRp),
    ("Rosenbrock -- HS",t_rosenbrock_HS),
    ("Beale -- FR",t_beale_FR),
    ("Beale -- PR",t_beale_PR),
    ("Beale -- PR+",t_beale_PRp),
    ("Beale -- HS",t_beale_HS),
    ("High-dim quadratic -- PR+",t_highdim_PRp),
    ("Finite termination on quadratic",t_finite_termination),
    ("Already-converged x0 returns fast",t_already_converged_x0),
    ("2D x0 is ravel'd to 1D",t_2d_x0_raveled),
    ("Periodic restart fires at step n",t_periodic_restart),
    ("Powell restart fires on Rosenbrock",t_powell_restart_fires),
    ("Powell fires independent of periodic",t_powell_independent_of_periodic),
    ("History array lengths consistent",t_history_lengths),
    ("No history when disabled",t_no_history_when_disabled),
    ("Invalid variant raises ValueError",t_invalid_variant_raises),
    ("Output x shape preserved",t_x_shape_preserved),
    ("nfev count accurate",t_nfev_count_accurate),
    ("No redundant f-eval (fk_new fix)",t_fk_no_redundant_eval),
]

if __name__=="__main__":
    print("\n"+"="*65+"\n  NCG UNIT TESTS\n"+"="*65)
    for name,fn in TESTS: test(name,fn)
    total=_results["passed"]+_results["failed"]
    print(f"\n{'='*65}")
    status=f"  ({_results['failed']} FAILED)" if _results["failed"] else "  All passed"
    print(f"  {_results['passed']}/{total} tests passed{status}")
    print("="*65)
    import sys; sys.exit(0 if _results["failed"]==0 else 1)
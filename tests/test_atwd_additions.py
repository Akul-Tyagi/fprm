"""Extra correctness tests for the reviewer-requested control arms. Run AFTER patches/apply_scalar_refresh_patch.py:
    python tests/test_atwd_additions.py
Test 5: scalar solver + refresh really refreshes (the control arm is not a silent baseline).
Test 6: scalar solver with refresh_interval=0 never resets step-size (baseline unchanged).
Test 7: with refresh, applied step-size never drops below delta**floor((R-1)/P)  (paper's 'bounded damping depth' remark),
        for BOTH the scalar and the token-wise solver.
Test 8: no-decay arm (delta=1.0) keeps step-size exactly eta0 forever.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from test_atwd import BASE_KW, run_steps
from models.fixed_point_reasoning.fprm_config import FPRMConfig
from models.fixed_point_reasoning.model_utils import FixedPointOptimizer


def make(token_wise, R=0, decay=0.5, patience=2, training=True):
    kw = dict(BASE_KW, stepsize_decay_train=decay, stepsize_decay_eval=decay, decay_patience=patience)
    opt = FixedPointOptimizer(FPRMConfig(**kw, token_wise_damping=token_wise, refresh_interval=R))
    opt.train(training)
    return opt


def test_5_scalar_refresh_resets():
    opt = make(False, R=5)           # patience=2 => first decay at iter 3, i.e. strictly between refreshes
    state, tr = run_steps(opt, n_steps=16, seed=2)
    assert state["stepsize"].shape == (2, 1, 1), state["stepsize"].shape
    assert any((ss < 1.0).any() for ss in tr["stepsize"]), "test is vacuous: no decay ever happened"
    for i, ss in enumerate(tr["stepsize"]):
        if (i + 1) % 5 == 0:
            assert torch.allclose(ss, torch.ones_like(ss)), f"scalar refresh missing at iter {i+1}: {ss.flatten()}"
    print("Test 5 (scalar solver + refresh resets step-size) PASSED")


def test_6_scalar_no_refresh_monotone():
    opt = make(False, R=0)
    state, tr = run_steps(opt, n_steps=12, seed=2)
    seq = torch.stack([s.flatten() for s in tr["stepsize"]])      # (iters, B)
    assert (seq[1:] <= seq[:-1] + 1e-9).all(), "step-size increased although refresh_interval=0"
    print("Test 6 (scalar, refresh_interval=0 never resets) PASSED")


def test_7_depth_bound():
    for token_wise in (False, True):
        for (decay, P, R) in [(0.5, 2, 6), (0.5, 3, 8), (0.9, 5, 20), (0.95, 10, 20)]:
            bound = decay ** ((R - 1) // P)
            for seed in range(6):
                opt = make(token_wise, R=R, decay=decay, patience=P)
                state, tr = run_steps(opt, n_steps=3 * R + 1, seed=seed)
                lo = min(float(ss.min()) for ss in tr["stepsize"])
                assert lo >= bound - 1e-6, f"bound violated tw={token_wise} d={decay} P={P} R={R} seed={seed}: {lo} < {bound}"
    print("Test 7 (applied step-size >= delta^floor((R-1)/P) under refresh) PASSED")


def test_8_no_decay_constant():
    for token_wise in (False, True):
        opt = make(token_wise, R=0, decay=1.0)
        state, tr = run_steps(opt, n_steps=20, seed=1)
        assert all(torch.allclose(ss, torch.ones_like(ss)) for ss in tr["stepsize"])
    print("Test 8 (no-decay arm keeps step-size = eta0) PASSED")


if __name__ == "__main__":
    test_5_scalar_refresh_resets()
    test_6_scalar_no_refresh_monotone()
    test_7_depth_bound()
    test_8_no_decay_constant()
    print("\nALL ADDITIONAL TESTS PASSED")

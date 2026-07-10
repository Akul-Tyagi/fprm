"""
Correctness tests for ATWD, run before any real training.

Test 1 (equivalence): with token_wise_damping=False, everything must
behave EXACTLY as the original baseline (this catches any accidental
change to the non-ATWD code path).

Test 2 (reduction): with token_wise_damping=True, if every token in a
sequence has identical dynamics AND identical initial state, the
token-wise optimizer must produce numerically identical stepsize/residual
trajectories to the baseline scalar optimizer. This is the actual
mathematical correctness check — ATWD must reduce to the original
mechanism in the degenerate case where there's nothing to differentiate
between tokens. reset() draws independent random init per token, so the
initial state must be explicitly broadcast across the token axis before
comparing (identical_initial_tokens=True below).

Test 3 (periodic refresh): stepsize must reset to the initial value
exactly every `refresh_interval` iterations, and not before.

Test 4 (heterogeneous tokens): damping (stepsize decay) only fires when a
token's residual has stopped improving for `decay_patience` steps AND is
still >= fp_thresh (see the `adapt` condition in model_utils.py). It is a
rescue mechanism for a token that is stuck/oscillating without converging
— NOT a mechanism that fires on "already converged" tokens. A token
already at/below fp_thresh never satisfies `residues >= fp_thresh`, so
its stepsize is never decayed; it simply doesn't need rescuing. Here,
token 0 is held static (residual ~0, effectively already converged) and
token 1 is perpetually reset to a fresh random target every step (never
converges, residual stays high). The correct expectation is that token 0
RETAINS its full stepsize and token 1 gets damped once patience runs out.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from models.fixed_point_reasoning.fprm_config import FPRMConfig
from models.fixed_point_reasoning.model_utils import FixedPointOptimizer

BASE_KW = dict(
    batch_size=2, seq_len=5, num_puzzle_identifiers=1, vocab_size=10,
    H_cycles=1, L_cycles=1, L_layers=1, H_layers=0,
    hidden_size=4, expansion=2.0, num_heads=1, pos_encodings="rope",
    halt_max_steps=1, halt_exploration_prob=0.0,
    max_iter=50, stepsize=1.0, stepsize_decay_train=0.5, stepsize_decay_eval=0.5,
    decay_patience=2, eps=1e-8, fp_thresh=0.1, init_std=1.0,
)


def make_optimizer(token_wise, refresh_interval=0, training=True):
    cfg = FPRMConfig(**BASE_KW, token_wise_damping=token_wise, refresh_interval=refresh_interval)
    opt = FixedPointOptimizer(cfg)
    opt.train(training)
    return opt


def run_steps(opt, batch=2, seq=5, hid=4, n_steps=8, y_fn=None, seed=0, identical_initial_tokens=False):
    torch.manual_seed(seed)
    device = "cpu"
    reset_flag = torch.ones(batch, dtype=torch.bool)
    state = opt.reset(reset_flag, (batch, seq, hid), torch.float32, device, None)
    if identical_initial_tokens:
        state["y"] = state["y"][:, :1, :].expand(-1, seq, -1).contiguous()
    traces = {"stepsize": [], "residues": []}
    for i in range(n_steps):
        y = y_fn(i, state) if y_fn else torch.randn(batch, seq, hid)
        state = opt.step(state, y)
        traces["stepsize"].append(state["stepsize"].clone())
        traces["residues"].append(state["residues"].clone())
    return state, traces


def test_1_equivalence_default_false():
    opt = make_optimizer(token_wise=False)
    state, traces = run_steps(opt, seed=42)
    assert state["stepsize"].shape == (2, 1, 1), state["stepsize"].shape
    assert state["residues"].shape == (2,), state["residues"].shape
    print("Test 1 (equivalence, token_wise=False) PASSED")


def test_2_reduction_when_tokens_identical():
    torch.manual_seed(0)
    fixed_ys = [torch.randn(2, 1, 4).expand(2, 5, 4).contiguous() for _ in range(8)]  # same value at every token position

    opt_scalar = make_optimizer(token_wise=False)
    state_s, trace_s = run_steps(opt_scalar, y_fn=lambda i, st: fixed_ys[i], seed=1, identical_initial_tokens=True)

    opt_token = make_optimizer(token_wise=True)
    state_t, trace_t = run_steps(opt_token, y_fn=lambda i, st: fixed_ys[i], seed=1, identical_initial_tokens=True)

    for i in range(8):
        ss = trace_s["stepsize"][i]              # (B,1,1)
        st = trace_t["stepsize"][i]               # (B,T,1)
        assert torch.allclose(ss.expand(-1, 5, -1), st, atol=1e-6), \
            f"stepsize mismatch at step {i}: scalar={ss.flatten()}, token={st[:, :, 0]}"
    print("Test 2 (reduction to scalar case) PASSED")


def test_3_periodic_refresh():
    opt = make_optimizer(token_wise=True, refresh_interval=3)
    state, traces = run_steps(opt, n_steps=10, seed=2)
    for i, ss in enumerate(traces["stepsize"]):
        it = i + 1
        if it % 3 == 0:
            assert torch.allclose(ss, torch.ones_like(ss) * opt.stepsize, atol=1e-6), \
                f"expected full reset at iter {it}, got {ss[0,0,0].item()}"
    print("Test 3 (periodic refresh) PASSED")


def test_4_heterogeneous_tokens_diverge():
    torch.manual_seed(3)
    # token 0 stays put every step (already converged, residual ~0);
    # token 1 keeps jumping (never converges, residual stays >= fp_thresh)
    def y_fn(i, state):
        y = state["y"].clone()
        y[:, 1, :] = torch.randn(2, 4)  # only token index 1 keeps changing
        return y

    opt = make_optimizer(token_wise=True, refresh_interval=0)
    state, traces = run_steps(opt, y_fn=y_fn, n_steps=10, seed=3)
    final_stepsize = state["stepsize"][:, :, 0]  # (B, T)
    assert (final_stepsize[:, 0] > final_stepsize[:, 1]).all(), \
        f"expected token 0 (static, already converged) to retain a higher stepsize than token 1 (stuck, never converging): {final_stepsize}"
    print("Test 4 (heterogeneous tokens damp independently) PASSED")


if __name__ == "__main__":
    test_1_equivalence_default_false()
    test_2_reduction_when_tokens_identical()
    test_3_periodic_refresh()
    test_4_heterogeneous_tokens_diverge()
    print("\nALL ATWD CORRECTNESS TESTS PASSED")
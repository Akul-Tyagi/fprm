"""Lets the periodic refresh run in the SCALAR (non-token-wise) solver too.

Why: control arm "scalar damping + refresh (R=20)" requested by reviewer 4Uap.
Before this patch, FixedPointOptimizer.step() only refreshes when token_wise=True, so
`arch.refresh_interval=20` with token_wise_damping=false would be SILENTLY IGNORED
(the run would just be a second baseline). Baseline runs (refresh_interval=0) are
bit-for-bit unchanged.

Usage (from repo root):  python patches/apply_scalar_refresh_patch.py
Safe to re-run: it refuses if the file is already patched.
"""
import re, sys
from pathlib import Path

P = Path("models/fixed_point_reasoning/model_utils.py")
src = P.read_text()

OLD = """        if self.token_wise and self.refresh_interval > 0:
            do_refresh = (state['iter_idx'] % self.refresh_interval == 0)
            state['stepsize'] = torch.where(
                do_refresh.view(-1, 1, 1),
                self.stepsize * torch.ones_like(state['stepsize']),
                state['stepsize'],
            )
            state['patience'] = torch.where(
                do_refresh.view(-1, 1),
                self.decay_patience * torch.ones_like(state['patience']),
                state['patience'],
            )
            state['best_residues'] = torch.where(
                do_refresh.view(-1, 1),
                torch.full_like(state['best_residues'], torch.inf),
                state['best_residues'],
            )
"""
NEW = """        if self.refresh_interval > 0:
            do_refresh = (state['iter_idx'] % self.refresh_interval == 0)          # (B,)
            m3 = do_refresh.view(-1, 1, 1)                                        # matches stepsize (B,T,1) or (B,1,1)
            m2 = do_refresh.view(-1, 1) if self.token_wise else do_refresh        # matches patience/best (B,T) or (B,)
            state['stepsize'] = torch.where(
                m3,
                self.stepsize * torch.ones_like(state['stepsize']),
                state['stepsize'],
            )
            state['patience'] = torch.where(
                m2,
                self.decay_patience * torch.ones_like(state['patience']),
                state['patience'],
            )
            state['best_residues'] = torch.where(
                m2,
                torch.full_like(state['best_residues'], torch.inf),
                state['best_residues'],
            )
"""
if "m3 = do_refresh.view(-1, 1, 1)" in src:
    sys.exit("Already patched - nothing to do.")
pat = r"\s+".join(re.escape(w) for w in OLD.split())
ms = list(re.finditer(pat, src))
assert len(ms) == 1, f"expected exactly 1 match of the old refresh block, found {len(ms)}; file differs from expected - do NOT patch by hand blindly"
src = src[:ms[0].start()] + NEW.strip() + src[ms[0].end():]   # match starts at "if" (indent already in file), so strip NEW's leading indent
P.write_text(src)
print("Patched", P)

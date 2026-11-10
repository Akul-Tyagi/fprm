"""Rebuild a trained run exactly the way pretrain.py does, load a plain `step_N` weights file, evaluate.

Why not create_model.load_checkpoint?  It unconditionally reads model.model.puzzle_emb.weights, which does not exist for the
state-tracking models (puzzle_emb_ndim=0) -> AttributeError.  We therefore load the weights ourselves.
"""
import os, re, sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import torch.distributed as dist


def init_dist():
    """Same pattern as pretrain.launch(): works with plain `python` (1 GPU) and `torchrun` (n GPUs)."""
    rank, world, cpu_group = 0, 1, None
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        cpu_group = dist.new_group(backend="gloo")
    return rank, world, cpu_group


def find_plain_ckpt(ckpt_dir):
    """Highest-numbered plain `step_<N>` file (EMA weights for EMA runs). Ignores *_train_state.pt and *_all_preds.* files."""
    best = None
    for f in Path(ckpt_dir).iterdir():
        m = re.fullmatch(r"step_(\d+)", f.name)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), f)
    if best is None:
        raise FileNotFoundError(f"no plain step_<N> weights file in {ckpt_dir}")
    return best[1]


def build_model(config_name, overrides, rank, world):
    """Returns (config, model).  `overrides` are ordinary Hydra overrides, e.g. ['data_paths=[data/x]', 'arch.token_wise_damping=true']."""
    from hydra import initialize_config_dir, compose
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    from pretrain_config import PretrainConfig
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
    from create_model import create_model
    config = PretrainConfig(**cfg)
    config.checkpoint_path = None
    config.load_checkpoint = None            # we load weights ourselves (see module docstring)
    train_meta = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed, dataset_paths=config.data_paths, global_batch_size=config.global_batch_size,
        test_set_mode=False, epochs_per_iter=1, rank=rank, num_replicas=world), split="train").metadata
    model, _opts, _lrs = create_model(config, train_meta, rank=rank, world_size=world)
    return config, model


def load_weights(model, path):
    sd = torch.load(str(path), map_location="cuda")
    has_prefix = any(k.startswith("_orig_mod.") for k in model.state_dict().keys())

    def fix(k):
        if has_prefix and not k.startswith("_orig_mod."):
            return "_orig_mod." + k
        if not has_prefix and k.startswith("_orig_mod."):
            return k[len("_orig_mod."):]
        return k
    res = model.load_state_dict({fix(k): v for k, v in sd.items()}, strict=False)
    bad = [k for k in res.missing_keys if not k.endswith(".mask")]     # same tolerance as utils/resume.py
    assert not bad and not res.unexpected_keys, f"checkpoint/model mismatch: missing={bad[:5]} unexpected={list(res.unexpected_keys)[:5]}"
    return model


def test_loader(config, rank, world):
    from pretrain import create_dataloader
    return create_dataloader(config, "test", rank=rank, world_size=world, test_set_mode=True,
                             epochs_per_iter=1, global_batch_size=config.global_batch_size)


def run_native_eval(config, model, rank, world, cpu_group):
    """Exactly pretrain.evaluate() (same halting, same batching, same length-gen accumulation)."""
    from pretrain import evaluate
    from utils.flop_profiler import FlopProfiler
    loader, meta = test_loader(config, rank, world)
    model.eval()
    ts = SimpleNamespace(model=model, step=0)
    return evaluate(config, ts, loader, meta, [], rank=rank, world_size=world, cpu_group=cpu_group, profiler=FlopProfiler(enabled=False))

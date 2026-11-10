"""Evaluate a trained checkpoint with an arbitrary SOLVER setting (cross-evaluation), using the repo's own evaluate().

Answers reviewer questions "is the failure in the weights or in the solver?":
  * ATWD-trained weights + SCALAR solver     (arch.token_wise_damping=false arch.refresh_interval=0)
  * baseline-trained weights + ATWD solver   (arch.token_wise_damping=true  arch.refresh_interval=20)
  * native (same solver as training) = sanity check: must reproduce the W&B final eval numbers.

State tracking example (run under torchrun with 2 GPUs so each rank sees the same 416-example batches as in training):
  torchrun --standalone --nproc-per-node=2 scripts/cross_eval.py \
     --config-name cfg_pretrain_state_tracking_fprm_a5 --ckpt-dir checkpoints/a5-atwd-r20-seed1 \
     --tag a5_s1_atwdW_scalarSolver --out results/xeval/a5_s1_atwdW_scalarSolver.json \
     data_paths=[data/state_tracking-A5-train32-eval128-500k] global_batch_size=832 k_eval_step=1 \
     arch.token_wise_damping=false arch.refresh_interval=0
Everything after the named flags is passed to Hydra as overrides.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_utils as U
import torch.distributed as dist
import torch


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    try:
        return float(x)
    except Exception:
        return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    args, overrides = ap.parse_known_args()
    rank, world, cpu = U.init_dist()
    config, model = U.build_model(args.config_name, overrides, rank, world)
    ckpt = U.find_plain_ckpt(args.ckpt_dir)
    if rank == 0:
        print(f"[cross_eval] tag={args.tag} ckpt={ckpt} world={world} solver: token_wise={config.arch.__pydantic_extra__.get('token_wise_damping')} "
              f"refresh={config.arch.__pydantic_extra__.get('refresh_interval')}", flush=True)
    U.load_weights(model, ckpt)
    metrics = U.run_native_eval(config, model, rank, world, cpu)
    # 4UAP CATCH: Save the final step-size distribution from the optimizer state
    if rank == 0 and hasattr(model.inner.L_optimizer, 'state') and 'stepsize' in model.inner.L_optimizer.state:
        stepsize_tensor = model.inner.L_optimizer.state['stepsize'].detach().cpu()
        torch.save(stepsize_tensor, str(Path(args.out).with_suffix('.stepsize.pt')))
    if rank == 0:
        out = {"tag": args.tag, "ckpt": str(ckpt), "overrides": overrides, "metrics": jsonable(metrics)}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
        m = metrics.get("all", {})
        print(f"[cross_eval] DONE {args.tag}: accuracy={m.get('accuracy')} sequence_accuracy={m.get('sequence_accuracy')} steps={m.get('steps')} -> {args.out}", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

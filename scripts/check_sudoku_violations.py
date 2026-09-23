import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import argparse
import glob
from tqdm import tqdm
from omegaconf import OmegaConf

from models.fixed_point_reasoning.fprm import FixedPointReasoningModel_ACTV1

def sudoku_violations(grid_9x9):
    v = 0
    for i in range(9):
        row = grid_9x9[i, :]; v += 9 - len(set(row.tolist()))
        col = grid_9x9[:, i]; v += 9 - len(set(col.tolist()))
    for br in range(3):
        for bc in range(3):
            box = grid_9x9[br*3:br*3+3, bc*3:bc*3+3].flatten()
            v += 9 - len(set(box.tolist()))
    return v

def load_model(checkpoint_dir, device, sample_batch):
    config_path = os.path.join(checkpoint_dir, "all_config.yaml")
    assert os.path.exists(config_path), f"Config not found at {config_path}"
    config = OmegaConf.load(config_path)
    arch_dict = OmegaConf.to_container(config.arch, resolve=True)
    
    # Target the EMA weights
    all_cands = glob.glob(os.path.join(checkpoint_dir, "step_*"))
    cands = [p for p in all_cands if not p.endswith("_train_state.pt") and not p.endswith(".yaml")]
    assert cands, f"No EMA checkpoint found in {checkpoint_dir}"
    latest = max(cands, key=lambda p: int(os.path.basename(p).replace('.pt', '').split("_")[1]))
    
    print(f"\n--- AUDIT METADATA ---")
    print(f"Loaded Checkpoint: {latest} (EMA Selection Confirmed)")
    print(f"Runtime token_wise_damping: {arch_dict.get('token_wise_damping', False)}")
    print(f"Runtime refresh_interval: {arch_dict.get('refresh_interval', 0)}")
    
    raw_sd = torch.load(latest, map_location=device)
    
    arch_dict["batch_size"] = sample_batch["inputs"].shape[0]
    arch_dict["seq_len"] = sample_batch["inputs"].shape[1]
    
    for k, v in raw_sd.items():
        if v.ndim == 2:
            if "puzzle" in k.lower() and "emb" in k.lower():
                arch_dict["num_puzzle_identifiers"] = v.shape[0]
            elif ("token" in k.lower() or "word" in k.lower() or "embed" in k.lower()) and "pos" not in k.lower():
                if v.shape[0] < 50:
                    arch_dict["vocab_size"] = v.shape[0]

    if "vocab_size" not in arch_dict:
        arch_dict["vocab_size"] = getattr(config, "vocab_size", 20)
    if "num_puzzle_identifiers" not in arch_dict:
        arch_dict["num_puzzle_identifiers"] = getattr(config, "num_puzzle_identifiers", 2)
    
    model = FixedPointReasoningModel_ACTV1(arch_dict).to(device).to(torch.bfloat16)
    
    prefix = "_orig_mod.model."
    sd = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in raw_sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/sudoku-extreme-1k-aug1k")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inputs_np = np.load(os.path.join(args.data_path, "test", "all__inputs.npy"))[:800]
    pids_np = np.load(os.path.join(args.data_path, "test", "all__puzzle_identifiers.npy"))[:800]
    
    inputs_tensor = torch.from_numpy(inputs_np).to(device)
    pids_tensor = torch.from_numpy(pids_np).to(device)
    
    total_puzzles = inputs_tensor.shape[0]
    batch_size = 32
    
    sample_batch = {
        "inputs": inputs_tensor[:batch_size],
        "puzzle_identifiers": pids_tensor[:batch_size]
    }
    
    model = load_model(args.checkpoint, device, sample_batch)

    print(f"Test Subset Puzzles: {total_puzzles}")
    print(f"Batch Size: {batch_size}")
    print(f"----------------------\n")

    total_violations = 0
    solved_count = 0
    failed_violations = 0

    with torch.inference_mode():
        for i in tqdm(range(0, total_puzzles, batch_size)):
            model_inputs = {
                "inputs": inputs_tensor[i:i+batch_size],
                "puzzle_identifiers": pids_tensor[i:i+batch_size]
            }
            
            current_b_size = model_inputs["inputs"].shape[0]
            reset_flag = torch.ones(current_b_size, dtype=torch.bool, device=device)
            empty_c = model.inner.empty_carry(current_b_size)
            carry = model.inner.reset_carry(reset_flag, model_inputs["inputs"], empty_c)
            
            _, logits, _ = model.inner(
                carry=carry, 
                batch=model_inputs, 
                force_grad=False, 
                n_steps=1000
            )
            
            preds = logits.argmax(dim=-1).cpu().numpy()
            if preds.shape[1] > 81:
                preds = preds[:, -81:]
                
            B, T = preds.shape
            assert T == 81, f"Expected 81 tokens for Sudoku grid, got {T}"
            
            for b in range(B):
                grid = preds[b].reshape(9, 9) + 1 
                v = sudoku_violations(grid)
                total_violations += v
                if v == 0:
                    solved_count += 1
                else:
                    failed_violations += v

    mean_violations = total_violations / total_puzzles
    failed_count = total_puzzles - solved_count
    mean_failed_violations = failed_violations / failed_count if failed_count > 0 else 0

    print(f"\n--- FINAL RESULTS: {args.checkpoint} ---")
    print(f"Total Duplicate Violations: {total_violations}")
    print(f"Exact Solved Count: {solved_count}")
    print(f"Overall Mean Violations: {mean_violations:.4f}")
    print(f"Mean Violations Among Failures: {mean_failed_violations:.4f}\n")

if __name__ == "__main__":
    main()
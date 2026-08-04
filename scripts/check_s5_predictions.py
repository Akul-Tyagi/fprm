"""
Inspection script: replays the fixed-point iteration for S5 ATWD Seed 1
and checks the predicted output token distribution.
"""
import glob
import json
import os
import numpy as np
import torch
from models.fixed_point_reasoning.fprm import FixedPointReasoningModel_ACTV1


def load_model(checkpoint_dir, config_dict, device):
    model = FixedPointReasoningModel_ACTV1(config_dict)
    cands = glob.glob(os.path.join(checkpoint_dir, "step_*_train_state.pt"))
    assert cands, f"No checkpoint bundle found in {checkpoint_dir}"
    latest = max(cands, key=lambda p: int(os.path.basename(p).split("_")[1]))
    bundle = torch.load(latest, map_location=device)
    raw_sd = bundle["model"]
    prefix = "_orig_mod.model."
    sd = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in raw_sd.items()}
    model.load_state_dict(sd, strict=False)
    
    # Cast to bfloat16 to match the checkpoint's precision
    model = model.to(device=device, dtype=torch.bfloat16) 
    model.eval()
    return model


def main():
    checkpoint_dir = "checkpoints/s5-atwd-r20-seed1"
    data_dir = "data/state_tracking-S5"
    config_json = "debug_arch_config_s5.json"
    num_examples = 4
    max_iter = 160
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(config_json) as f:
        config_dict = json.load(f)

    model = load_model(checkpoint_dir, config_dict, device)
    inner = model.inner

    inputs_np = np.load(os.path.join(data_dir, "test", "all__inputs.npy"))[:num_examples]
    labels_np = np.load(os.path.join(data_dir, "test", "all__labels.npy"))[:num_examples]
    puzzles_np = np.load(os.path.join(data_dir, "test", "all__puzzle_identifiers.npy"))[:num_examples]

    inputs = torch.from_numpy(inputs_np).to(device)
    puzzles = torch.from_numpy(puzzles_np).to(device)

    # Wrap the inference in an autocast block to ensure dtype safety
    device_type = "cuda" if "cuda" in device else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16), torch.no_grad():
        reset_flag = torch.ones(inputs.shape[0], dtype=torch.bool, device=device)
        carry = inner.reset_carry(reset_flag, inputs, inner.empty_carry(inputs.shape[0]))

        input_embeddings = inner._input_embeddings(inputs, puzzles)
        cos_sin = None
        if hasattr(inner, "rotary_emb"):
            cos, sin = inner.rotary_emb()
            s = input_embeddings.shape[1]
            cos_sin = (cos[:s], sin[:s])
        seq_info = dict(cos_sin=cos_sin, puzzle_emb_len=inner.puzzle_emb_len)

        z_state = carry.z_L_state
        for _ in range(max_iter):
            z_state = inner._z_step(z_state, input_embeddings, carry.dropout_mask, seq_info)

        # FIX: Extract the state from the correct key "y"
        hidden = z_state["y"]
        logits = inner.lm_head(hidden)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

    print("\n" + "=" * 60)
    print("S5 ATWD SEED 1 PREDICTION ANALYSIS")
    print("=" * 60)

    for i in range(num_examples):
        p = preds[i]
        l = labels_np[i]
        unique_tokens, counts = np.unique(p, return_counts=True)
        top_token = unique_tokens[np.argmax(counts)]
        top_freq = np.max(counts) / len(p) * 100

        print(f"\n--- Example {i} ---")
        print(f"Target Labels (first 20):   {l[:20].tolist()}")
        print(f"Predicted Tokens (first 20): {p[:20].tolist()}")
        print(f"Unique Predicted Tokens:    {len(unique_tokens)} distinct values: {unique_tokens.tolist()}")
        print(f"Most Frequent Token:        Token {top_token} ({top_freq:.1f}% of sequence)")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
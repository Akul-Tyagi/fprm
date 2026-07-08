## Dataset generation commands (for exact reproducibility)
uv run python dataset/build_state_tracking_dataset.py --group A5 --k-train 32 --k-eval-max 128 --train-samples 100000 --test-samples 1000 --seed 0 --output-dir data/state_tracking-A5
uv run python dataset/build_state_tracking_dataset.py --group S5 --k-train 32 --k-eval-max 128 --train-samples 100000 --test-samples 1000 --seed 0 --output-dir data/state_tracking-S5

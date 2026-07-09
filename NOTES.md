## Dataset generation commands (for exact reproducibility)
uv run python dataset/build_state_tracking_dataset.py --group A5 --k-train 32 --k-eval-max 128 --train-samples 100000 --test-samples 1000 --seed 0 --output-dir data/state_tracking-A5
uv run python dataset/build_state_tracking_dataset.py --group S5 --k-train 32 --k-eval-max 128 --train-samples 100000 --test-samples 1000 --seed 0 --output-dir data/state_tracking-S5
## Dataset generation commands
# Smoke-test sets (Section F):
uv run python dataset/build_state_tracking_dataset.py --group A5 --k-train 32 --k-eval-max 128 --train-samples 5000 --test-samples 200 --seed 0 --output-dir data/state_tracking-A5-smoketest
uv run python dataset/build_state_tracking_dataset.py --group S5 --k-train 32 --k-eval-max 128 --train-samples 5000 --test-samples 200 --seed 0 --output-dir data/state_tracking-S5-smoketest
# Full-scale sets (Section J) — matches cfg_pretrain_state_tracking_fprm_a5/s5.yaml's data_paths exactly:
uv run python dataset/build_state_tracking_dataset.py --group A5 --k-train 32 --k-eval-max 128 --train-samples 2000000 --output-dir data/state_tracking-A5-train32-eval128-2M
uv run python dataset/build_state_tracking_dataset.py --group S5 --k-train 32 --k-eval-max 128 --train-samples 2000000 --output-dir data/state_tracking-S5-train32-eval128-2M

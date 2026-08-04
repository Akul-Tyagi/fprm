# scripts/truncate_sudoku_test.py
import numpy as np, json, os, sys

def truncate_sudoku_test(data_dir, n_test=400):
    test_dir = os.path.join(data_dir, "test")
    meta_path = os.path.join(test_dir, "dataset.json")
    with open(meta_path) as f:
        meta = json.load(f)

    inputs      = np.load(os.path.join(test_dir, "all__inputs.npy"))
    labels      = np.load(os.path.join(test_dir, "all__labels.npy"))
    puzzle_ids  = np.load(os.path.join(test_dir, "all__puzzle_identifiers.npy"))
    puzzle_idx  = np.load(os.path.join(test_dir, "all__puzzle_indices.npy"))
    group_idx   = np.load(os.path.join(test_dir, "all__group_indices.npy"))

    n_test = min(n_test, len(group_idx) - 1)
    last_example = int(group_idx[n_test])

    np.save(os.path.join(test_dir, "all__inputs.npy"), inputs[:last_example])
    np.save(os.path.join(test_dir, "all__labels.npy"), labels[:last_example])
    np.save(os.path.join(test_dir, "all__puzzle_identifiers.npy"), puzzle_ids[:last_example])
    np.save(os.path.join(test_dir, "all__puzzle_indices.npy"), puzzle_idx[:n_test + 1])
    np.save(os.path.join(test_dir, "all__group_indices.npy"), group_idx[:n_test + 1])

    meta["total_groups"] = n_test
    meta["total_puzzles"] = n_test
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"Truncated {data_dir}/test to {n_test} puzzles ({last_example} examples).")

if __name__ == "__main__":
    truncate_sudoku_test(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 400)
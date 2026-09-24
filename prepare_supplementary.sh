#!/usr/bin/env bash
# Builds FPRM_ICLR2027/ from your working fprm/ repo for ICLR 2027 supplementary
# material. Explicit include-list (not exclude-list) on purpose: an include-list
# fails safe — a file you forgot to add just doesn't show up, instead of a file
# you forgot to exclude silently riding along in the zip. Re-run any time; it
# overwrites the destination each time so it never accumulates stale copies.
#
# Usage:
#   chmod +x prepare_supplementary.sh
#   ./prepare_supplementary.sh /path/to/fprm /path/to/FPRM_ICLR2027

set -euo pipefail

SRC="${1:?Usage: $0 <source fprm dir> <destination FPRM_ICLR2027 dir>}"
DST="${2:?Usage: $0 <source fprm dir> <destination FPRM_ICLR2027 dir>}"

if [ ! -d "$SRC" ]; then echo "Source not found: $SRC"; exit 1; fi

rm -rf "$DST"
mkdir -p "$DST"

echo "== Copying code directories =="
for d in config dataset evaluators models plots scripts tests utils; do
  if [ -d "$SRC/$d" ]; then
    cp -r "$SRC/$d" "$DST/$d"
    echo "  copied $d/"
  else
    echo "  !! MISSING expected dir: $d (skipped)"
  fi
done

echo "== Copying root-level source files =="
for f in pretrain.py pretrain_config.py create_model.py puzzle_dataset.py \
         length_gen_eval.py pyproject.toml uv.lock fig1.png \
         debug_arch_config_a5.json debug_arch_config_s5.json; do
  if [ -f "$SRC/$f" ]; then
    cp "$SRC/$f" "$DST/$f"
    echo "  copied $f"
  else
    echo "  !! MISSING expected file: $f (skipped)"
  fi
done

echo "== Files needing your manual sign-off before inclusion =="
for f in LICENSE wandb_rundata.py; do
  if [ -f "$SRC/$f" ]; then
    cp "$SRC/$f" "$DST/$f.REVIEW_BEFORE_INCLUDING"
    echo "  copied $f -> $f.REVIEW_BEFORE_INCLUDING (open it, check for your name/username, then rename to drop the suffix once clean)"
  fi
done

echo "== Copying results/ (figures + analysis arrays), then checking size =="
if [ -d "$SRC/results" ]; then
  mkdir -p "$DST/results"
  find "$SRC/results" -maxdepth 1 -name "*.png" -exec cp {} "$DST/results/" \;
  [ -d "$SRC/results/figures" ] && cp -r "$SRC/results/figures" "$DST/results/figures"
  find "$SRC/results" -maxdepth 1 -name "*.npz" -exec cp {} "$DST/results/" \;
  echo "  copied results/*.png, results/figures/, results/*.npz"
fi

echo
echo "== NOT copied (by design — see README for why) =="
echo "  checkpoints/   (25+GB of model weights — far too large; not needed for review)"
echo "  data/          (generated .npy dataset arrays — hundreds of MB+; regenerate with the commands in README.md)"
echo "  outputs/       (Hydra per-run logs — may embed your machine hostname via os.uname())"
echo "  wandb/         (contains symlinks to /home/akul/.cache/... — a direct identity leak; NEVER include this folder)"
echo "  fprm.egg-info/ (auto-generated packaging metadata, often duplicates author/email from pyproject.toml)"
echo "  *.log at repo root (sudoku_ema_violations_audit.log, tea_debug.log — their conclusions are already in the paper's tables)"
echo "  NOTES.md, project_structure.txt (superseded by the new README.md)"
echo

echo "== Final size check =="
du -sh "$DST"
echo
echo "If results/*.npz pushes you over 100MB, the safe order to trim is:"
echo "  1. drop results/*.npz first (the .png figures are the important, tiny part)"
echo "  2. if still too big, keep only the 2-3 .npz files actually cited by name in the paper's appendix"
echo
echo "== Mandatory final safety net: run this over $DST before zipping =="
echo "  grep -rniE 'akul|tyagi|galgotia|/home/[a-z]+|kaggle\\.com/[a-z]' \"$DST\" | grep -v '\\.REVIEW_BEFORE_INCLUDING'"
echo "  (should print nothing; investigate any hit before zipping)"

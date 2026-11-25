#!/usr/bin/env bash
# Full experiment pipeline, in the order used for the paper.
# Requires data/lobster/<TICKER>/ (see data/README.md) and a CUDA GPU (CPU works but is far too slow).
# Roughly 10+ hours on one RTX 5070 laptop GPU. Every stage writes to results/ and can be run on its own.
# Usage:  bash scripts/run_pipeline.sh        (set PYTHON=/path/to/python to use a specific interpreter)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
mkdir -p results/models

echo "== 1. Baselines + 3-seed U-Nets (2/3/6 ch, BCE+Dice) + loss ablations -> results/main.json"
$PY code/run_main.py
echo "== 2. BCE-only 3-ch model (legacy tag unet3_bce)"
$PY code/train_bce_model.py
echo "== 3. Training grid: BCE-only 2/3/6 ch, decoy-trained models, four loss variants (3 seeds each) -> results/grid.json"
$PY code/train_grid.py

echo "== 4. Hard-negative test of models trained without decoys"
$PY code/eval_decoys.py

echo "== 5. Legacy single-seed models: diagnosis, sweeps, execution, out-of-generator, real-book checks"
$PY code/diagnose.py unet2 unet3 unet6
$PY code/run_sweeps.py unet3 unet6
$PY code/run_sweeps.py unet3_bce --bce-heldout
for tag in unet3 unet6 unet3_bce; do
  $PY code/execsim.py "$tag" 2500
  $PY code/run_robust.py ood "$tag"
  $PY code/run_robust.py real "$tag"
done
$PY code/diagnose.py unet3_bce

echo "== 6. Per-seed analyses of the BCE-only 2/3/6-channel models"
for base in m3b m6b m2b; do
  for k in 0 1 2; do
    tag="${base}@${k}"
    $PY code/diagnose.py "$tag"
    $PY code/run_sweeps.py "$tag"
    $PY code/execsim.py "$tag" 2500
    $PY code/run_robust.py ood "$tag"
    $PY code/run_robust.py real "$tag"
  done
done

echo "== 7. Decoy-trained models: execution, out-of-generator, real-book checks"
for base in m3b_hn m6b_hn; do
  for k in 0 1 2; do
    tag="${base}@${k}"
    $PY code/execsim.py "$tag" 2500
    $PY code/run_robust.py ood "$tag"
    $PY code/run_robust.py real "$tag"
  done
done

echo "== 8. Loss-variant diagnosis (per-asset precision, seed 0)"
for base in m3_dpo m3_bdb m3_pw m3_focal; do
  $PY code/diagnose.py "${base}@0"
done

echo "== 9. Execution sensitivity, time-bin ablation, weak labels, FI-2010 check, latency"
$PY code/run_exec_sens.py unet3_bce
$PY code/run_robust.py timecols
$PY code/weak_labels.py unet3_bce "m3b@0" "m3b_hn@0" "m6b_hn@0"
$PY code/fi2010.py "m2b@0"          # needs data/fi2010/raw/*.txt (optional, see data/README.md)
$PY code/measure_latency.py         # run on an otherwise idle GPU

echo "== 10. Tables and figures (written to paper/gen, paper/gen2, paper/figs)"
$PY code/make_tables.py
$PY code/make_tables2.py
$PY code/make_figures.py
$PY code/make_figures_extra.py
echo "Done."

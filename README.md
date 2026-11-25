# Financial Lane Detection

**Semantic segmentation of high-frequency limit order books for anomaly masking.**

A window of a limit order book (LOB) is treated as an image: rows are book levels, columns are order book events,
channels are liquidity and order-flow variables. A U-Net outputs a per-cell probability map marking liquidity that looks
spoof-like (large, short-lived, pulled before trading), so that downstream algorithms can discount it.

> **Scope, in one paragraph.** There are no public spoofing labels. Labels here come from _injecting_ spoof-like orders into
> real NASDAQ order books (LOBSTER sample data, one trading day, five stocks) together with _decoys_ (large genuine-looking
> orders labelled as normal). Every score therefore measures recovery of a defined, synthetic anomaly class. It is **not**
> evidence of detecting real market manipulation, and the project reports checks that point the other way (see Results).

**Published:** J. Gupta and M. Gupta, "Financial Lane Detection: Semantic Segmentation of High-Frequency Limit Order Books
for Anomaly Masking", _Telematique_, vol. 25, no. 1, pp. 3189-3205, 2026 (ISSN 1856-4194). The experiments reported in the
paper were produced with this code. See [Citation](#citation).

Authors: Jay Gupta and Dr. Meenakshi Gupta, School of Engineering & Technology, Sushant University.

## Main results (test windows, mean over 3 seeds)

| Model                                                    | IoU (standard test)   |
| -------------------------------------------------------- | --------------------- |
| Volume threshold / local z-score / CNN window classifier | 0.113 / 0.015 / 0.163 |
| U-Net, 2 channels (BCE)                                  | 0.664 +- 0.092        |
| U-Net, 3 channels (BCE)                                  | 0.866 +- 0.017        |
| U-Net, 6 channels (BCE)                                  | 0.944 +- 0.004        |

- **Hard negatives.** Models trained without decoys flag about 42-45% of decoy cells; trained with decoys, about 2-3.5%.
  On a test set that contains decoys, IoU is 0.605 (3 ch) and 0.713 (6 ch): the more realistic figure.
- **Loss function.** The original per-window BCE+Dice loss gave IoU 0.624 and a precision collapse on the two busiest
  stocks late in the day; BCE alone gives 0.866 and Dice restricted to windows containing an anomaly gives 0.847.
  The suspected cause (Dice over empty windows) is supported by this one comparison, not proven.
- **Negative/neutral findings, reported openly.** Overlap of model flags with real rule-based candidate orders is at chance
  level; the model flags 3-4% of FI-2010 cells (vs about 0.1% on clean LOBSTER); a stylised execution simulation shows a
  shortfall reduction of at most 0.79% over 15 settings.
- **A label leak was found and removed** (order-age channel); `tests/test_pipeline.py` guards against it.

## Repository layout

```
code/
  data.py           LOBSTER loader, level-event tensor, channels, splits, windows
  inject.py         spoof injector, decoys (hard negatives), out-of-generator spoof families
  models.py         U-Net, CNN baseline, volume-threshold / z-score detectors, loss variants
  modelio.py        load a saved model by tag ("unet3_bce" or "m3b@1")
  experiment.py     dataset building, training loop, metrics (IoU, Dice, AP, component recall, decoy-flag rate)
  run_main.py       baselines + 3-seed U-Nets (2/3/6 ch) + loss ablations
  train_grid.py     BCE-only 2/3/6 ch, decoy-trained models, four loss variants (3 seeds, models saved)
  train_bce_model.py  BCE-only 3-channel model used by several downstream stages
  run_sweeps.py     size / duration / volatility / per-asset sweeps, latency, held-out-asset run
  run_robust.py     out-of-generator test, unlabelled real-book check, time-bin ablation
  diagnose.py       precision-gap diagnosis, threshold analysis, channel zeroing
  eval_decoys.py    models trained without decoys, tested with decoys
  weak_labels.py    weak labels from real order lifecycles (LOBSTER order IDs)
  execsim.py        stylised execution simulation; run_exec_sens.py = sensitivity grid
  fi2010.py         external FI-2010 background-structure check
  measure_latency.py  batch-1 inference latency
  make_tables.py, make_tables2.py, make_figures.py, make_figures_extra.py   LaTeX tables / figures from results/
scripts/run_pipeline.sh   the full run, in the order used for the paper
tests/test_pipeline.py    pipeline and leak tests
data/README.md            where to put the data (no data is included)
```

`results/` (JSON, logs, saved models) and `paper/` (generated tables and figures) are created by the scripts and are not
tracked in git.

## Setup

```bash
pip install -r requirements.txt
```

Developed and tested on Python 3.13 (other versions untested). Training needs a CUDA GPU; the numbers in the paper were produced on an
RTX 50-series laptop GPU with PyTorch built for CUDA 12.8 (older builds do not support its compute capability).

Get the data as described in [`data/README.md`](data/README.md) (LOBSTER free samples; FI-2010 optional).

## Running

```bash
python tests/test_pipeline.py            # needs data/lobster/GOOG; runs in a minute or two, checks leak guards
python code/run_main.py                  # baselines + U-Nets, writes results/main.json
python code/train_grid.py                # several hours: models for the channel / decoy / loss studies, results/grid.json
bash scripts/run_pipeline.sh             # everything, in order (10+ hours on one GPU)
```

Each script can be run on its own; they read and write `results/`. A model is addressed by a tag: `unet3_bce` is the
legacy seed-0 file `results/unet3_bce_seed0.pt`, and `m3b@1` is `results/models/m3b_s1.pt` (see `code/modelio.py`).
`m2b/m3b/m6b` are BCE-only 2/3/6-channel models, `*_hn` were trained with decoys, `m3_dpo/m3_bdb/m3_pw/m3_focal` are loss variants.

## Method in brief

- **Tensor:** 20 rows (10 ask levels, 10 bid levels), 256 event columns, 6 channels (ask volume, bid volume, cancellation
  rate, arrival rate, execution rate, order age); volumes are log-transformed and scaled with training-split statistics only.
- **Splits:** chronological within the day, 60 / 15 / 25 with a 256-event gap; injection is run independently per split.
- **Injected spoofs:** size 2-20x local level volume, level 2-10, lifetime 5-100 events, 35% cancelled early on price moves,
  25% multi-level, about 3 per 1000 events. **Decoys:** executed or persistent (150-400 events) large orders, label 0.
- **Model:** U-Net (widths 32-64-128-256, 512 bottleneck, about 7.76 M parameters); AdamW (1e-3, weight decay 1e-4),
  batch 64, LR halving on plateau, early stopping on validation IoU (patience 10, max 50 epochs); best-validation checkpoint.

## Reproducibility notes

- Seeds are fixed in the scripts; GPU non-determinism still changes results slightly (a retrained seed 0 differed by about 0.007 IoU).
- Latency should be measured on an otherwise idle GPU.
- This repository was assembled from the working project after the experiments were run. The unit tests were run on a clean
  copy.

## Limitations

One trading day and five stocks; synthetic labels (no real spoofing labels); a stylised execution simulation; the held-out-asset
run uses one seed; the explanation of the Dice failure is not isolated. See the paper for details.

## Data and licence

LOBSTER sample data and FI-2010 are third-party data: follow their terms. No licence has been chosen for this code yet;
add one (for example MIT) before making the repository public if you want others to reuse it.

## Citation

If you use this code or its results, please cite the paper:

```bibtex
@article{gupta2026financial,
  author  = {Gupta, Jay and Gupta, Meenakshi},
  title   = {Financial Lane Detection: Semantic Segmentation of High-Frequency Limit Order Books for Anomaly Masking},
  journal = {Telematique},
  volume  = {25},
  number  = {1},
  pages   = {3189--3205},
  year    = {2026},
  issn    = {1856-4194}
}
```

# KLMS-LEP: localized dictionary pruning for kernel adaptive filters

Code for the paper *Localized Dictionary Pruning for Kernel Adaptive Filters under Partial Concept Drift* (C. Saide, submitted to *Signal Processing*).

KLMS-LEP tracks, for every dictionary atom, the exact change in recent squared error that its removal would cause, Δᵢ = wᵢ(2Aᵢ + wᵢBᵢ), in O(D) per step. A one-sided test prunes atoms whose removal would lower the recent error (atoms whose local concept has changed), and the same statistic replaces first-in-first-out eviction when the dictionary reaches its budget.

## Layout
- `code/kaf.py`: all filters (KLMS-Dict with FIFO, ADWIN+Restart, ADWIN+RR-Prune, FB-QKLMS, SW-KRLS, KRLS-T, KLMS-LEP)
- `code/streams.py`: synthetic partial-drift streams; `code/data_real.py`: EMG and CWRU streams (features from `code/original_experiment.py`)
- `code/bench.py`: tuning on pre-drift data and the 20-seed benchmark
- `code/run_synthetic.py`, `code/sweep.py`: synthetic runs, budget sweep and λ/z sensitivity
- `code/figures.py`, `code/make_tables.py`: regenerate `figures/` and `tables/` from `results/`
- `results/`: per-seed CSVs, learning curves and tuned parameters used in the paper

## Data
The raw datasets are not redistributed here. Put them under `data/`:
- `data/emg/EMG Physical Action Data Set/`: the UCI EMG Physical Action data set (https://archive.ics.uci.edu/dataset/213)
- `data/cwru/{97,105,118,130,169,209}.mat`: drive-end 12 kHz files from the Case Western Reserve University Bearing Data Center

## Reproduce
```
pip install -r requirements.txt
cd code
python bench.py tune_<b>      # tune on pre-drift data (seed 0)
python bench.py <b> 20        # 20 seeds; b in partial25, partial50, partial100, emg, cwru4, cwru, inversion
python sweep.py
python figures.py
python make_tables.py
```

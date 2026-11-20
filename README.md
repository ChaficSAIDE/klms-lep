# Revision for IEEE Access: KLMS-LEP

The original files (`/mnt/project-files/main.tex`, `fig*.pdf`) are unchanged.

## Contents
- `paper/main_access.tex`, `paper/main_access.pdf`: revised paper in the IEEE Access template (`ieeeaccess.cls`), 9 pages. The tables are generated and inlined between `% BEGIN/% END` markers.
- `code/`: experiments
  - `kaf.py`: all filters (KLMS-Dict, ADWIN+Restart, ADWIN+RR-Prune = old DriftDict, FB-QKLMS, KRLS-T, SW-KRLS, **KLMS-LEP**)
  - `streams.py`: partial-drift benchmark; `data_real.py`: EMG and CWRU streams (feature extraction from `original_experiment.py`)
  - `bench.py`: `python bench.py tune_<b>` (pre-drift tuning), then `python bench.py <b> 20`, for b in partial25/50/100, emg, cwru4, cwru, inversion
  - `sweep.py`: budget sweep and λ/z sensitivity; `figures.py`; `make_tables.py` (regenerates the tables and inlines them in the paper)
- `data/`: EMG Physical Action (UCI) and CWRU .mat files, taken from public GitHub mirrors. The EMG stream reproduces the original 15,651 windows and drift points exactly.
- `results/`: raw per-seed CSVs, curves and tuned parameters

## What changed from the SPL version
1. **New core method.** ADWIN + recency–relevance score + quota p is replaced by an exact per-atom *removal gain* Δᵢ = wᵢ(2Aᵢ + wᵢBᵢ), tracked in O(D) per step. A one-sided test (z = 1.645) prunes atoms whose removal would lower recent error. Δᵢ also replaces FIFO eviction. α and p are gone, and λ = 0.98 and z = 1.645 are fixed everywhere.
2. **Analysis.** Lemma 1 (stable atoms kept in expectation), Lemma 2 (reversed atoms flagged), and Proposition 1 (localized pruning beats a full reset when p_S > 0).
3. **Partial-drift benchmark** with drift extent ρ ∈ {0.25, 0.5, 1}, plus a real-data partial-drift stream (CWRU 4-class, only the inner-race class worsens).
4. **Baselines added:** FIFO at the same budget, FB-QKLMS, KRLS-T and SW-KRLS. The old method is kept as "ADWIN+RR-Prune".
5. **20 seeds**, mean ± sd, Wilcoxon tests, first-300 accuracy and recovery time, a budget sweep, an ablation, and a sensitivity study.
6. **Fixes.** CWRU now uses true inner-race severities (105→169→209; the old code used 105/118/130 = IR/ball/OR). Hyperparameters are tuned on pre-drift data only. Captions now match the drift times. The KRLS-T reference had wrong authors (now Van Vaerenbergh, Lázaro-Gredilla, Santamaría).

## Honest limitations (also stated in the paper)
- KRLS-T matches LEP on synthetic data and beats it on EMG post-drift accuracy (0.892 vs 0.880) and on CWRU-4. LEP's advantages are cost (~4× cheaper per step) and faster recovery.
- The binary CWRU stream and the old global-inversion stream show no drift effect for any method. They are reported briefly as negative cases.
- On EMG with D=160 the dictionary is still saturated, and LEP is 0.8 points below FIFO there.

## Before submitting
- Add the code repository URL (Data and Code Availability section).
- IEEE requires disclosure of AI-generated content in the acknowledgments. Add a sentence on how AI tools were used for code, experiments and text.
- Check the references against the publishers' records (all were written from memory and checked for plausibility only). In particular, check Chen & Dai 2024 (kept from the SPL version) and the page ranges.

## Signal Processing (Elsevier) version (target since 2026-11-19)
- `paper_elsevier/main_sigpro.tex` / `.pdf`: elsarticle `review` mode (single column, double spaced, line numbers), 30 pages including figures, tables and references. That is exactly the journal's limit for research articles. Fast Communications are limited to 10 such pages, so this paper goes in as a regular research article.
- `paper_elsevier/highlights.txt`: 5 highlights (each ≤ 85 characters), to upload as a separate file.
- Already included: CRediT statement, competing-interest declaration, data availability, and the generative-AI declaration. **Fill in the AI tool and the purpose** in the last one, and add the repository URL.
- The IEEE Access version is kept in `paper/`.

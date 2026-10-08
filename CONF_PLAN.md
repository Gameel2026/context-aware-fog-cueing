# Analysis plan: activity-specific cue thresholds for FoG at home (conference paper)

**Written after the task inventory (annotations only, `C0_task_inventory.csv`) and before any model was
trained or any controller was evaluated for this study.** Commit this file before running `conf_step1_analysis.py`.

## Data
- defog (Kaggle FoG contest, home visits), lower-back accelerometer, 45 patients, 148,871 windows of 0.5 s,
  625 FoG episodes (2 s merging rule), same conversion, causal filtering and 72 features as in our previous work.
- Activity context comes **only from the protocol task annotations (tasks.csv)**, never from the FoG-type labels.

## Activity groups (fixed from the task names, before any model output)
| Group | Protocol tasks |
|---|---|
| Rest | Rest1, Rest2 |
| Balance | MB1–MB13 (all MB items) |
| Turning | Turning-ST, Turning-DT, Turning-C |
| Complex gait | Hotspot1, Hotspot2 (+ -C), TUG-ST, TUG-DT, TUG-C |
| Straight walking | 4MW, 4MW-C |
Windows without a task (0.4%) use the global thresholds.

## Models (subject-grouped, 10 outer folds; nothing fitted on held-out patients)
1. **Cue-need model:** the 2-layer LSTM of our previous work (4 s of features, 2 epochs, same settings),
   retrained within defog. Training-patient probabilities for threshold selection come from an inner
   subject-grouped 3-fold split (out-of-fold), as in our previous work.
2. **Activity recogniser:** random forest (100 trees, balanced class weights) on the 72 features plus causal
   context (rolling means over 2 s and 4 s, rolling SD over 4 s), trained on the training patients' task-group
   labels; output smoothed causally by the most frequent class over the last 8 windows (4 s).

## Controllers (hysteresis: cue on when p ≥ θon, off when p < θoff; thresholds of the current window's group)
- **C-global:** one (θon, θoff) pair for all windows.
- **C-oracle:** one pair per activity group, applied with the annotated group (upper bound, not deployable).
- **C-predicted:** the same per-group pairs, applied with the recogniser's group (deployable).

**Threshold rule (identical for all controllers):** on the training patients' out-of-fold probabilities, choose the
pair in the grid θon 0.30–0.95 (step 0.05), θoff 0.10–θon (step 0.10) that maximises window-level F1 of the
cue-needed target, computed over all training windows (C-global) or over the windows of each group (per-group
controllers). **Groups with fewer than 10 FoG episodes in the training patients** (expected: Balance) get the cue
disabled (θon above 1). Rest is learned like the others if it has ≥ 10 episodes.

## Outcomes
- **Primary:** false alarms per hour (activation runs not overlapping [onset − 4 s, offset + 3 s]), per patient,
  C-predicted vs C-global, two-sided Wilcoxon signed-rank test.
- **Secondary:** C-oracle vs C-global (same test); episode coverage and timely activation (cue switched on within
  the 4 s before onset, eligible episodes) of every controller against a circular-shift surrogate (2,000 shifts);
  cue specificity; activity-recogniser accuracy and per-group recall; false alarms per hour by group.
- Holm correction across the two primary-family comparisons (predicted vs global, oracle vs global) for false
  alarms; coverage compared with the same Wilcoxon/Holm scheme.

## Criteria fixed in advance
- **Supported:** C-predicted produces fewer false alarms per hour than C-global (Holm-adjusted p < 0.05) and its
  episode coverage is not lower than C-global's by more than 5 percentage points (pooled).
- Timely activation is reported whatever its value; no claim of improved prediction unless it exceeds the
  surrogate reference.
- All results are reported whatever their direction.

## Scope statement for the paper
Home recordings collected under a structured protocol (not free living); context labels exist only for protocol tasks.

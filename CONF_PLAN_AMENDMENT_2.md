# Amendment 2 to CONF_PLAN.md: muting control, task-preserving surrogate, value of activity recognition

**Written before any analysis of CONF_PLAN.md or Amendment 1 was run on either dataset.** No result of this study
existed when this amendment was written. CONF_PLAN.md and Amendment 1 are unchanged: the primary outcome, its
Holm family (C-predicted vs C-global and C-oracle vs C-global) and the primary verdict are not altered. Everything
below is **secondary** and applies to defog and FoG-STAR unless stated otherwise.

## A. Muting control (controller C-gate)
**Question:** does any reduction in false alarms come from activity-specific thresholds, or only from switching the
cue off in activity groups where it is disabled?
- **C-gate:** the global (θon, θoff) pair of C-global in every window, except that the cue is disabled in exactly the
  groups that are disabled for C-predicted in the same fold; applied with the **predicted** activity group (same
  recogniser and smoothing as C-predicted). In defog this is expected to be Balance; in FoG-STAR, Sitting.
- **Test:** C-predicted vs C-gate, per patient, two-sided Wilcoxon signed-rank test on false alarms per hour and on
  episode coverage; Holm correction across these two tests.
- **Reading fixed in advance:** activity-specific thresholds add value beyond muting if C-predicted produces fewer
  false alarms per hour than C-gate (pooled) **and** its coverage is not lower than C-gate's by more than 5 percentage
  points **and** (defog only) the Holm-adjusted p for false alarms is < 0.05. For FoG-STAR the direction alone is read,
  as in Amendment 1.

## B. Task-preserving surrogate (defog only)
- In addition to the circular-shift surrogate of CONF_PLAN.md, the cue output of every controller is shifted
  circularly **within each contiguous block of the same protocol task** (windows without a task are kept fixed;
  2,000 shifts), so that the null distribution keeps the activity structure of the recordings.
- **Reading:** episode coverage and timely activation above this reference (one-sided p < 0.05) indicate that the cue
  follows the episodes themselves, not only the tasks in which freezing is frequent.
- Not applied to FoG-STAR: its activity segments last a few seconds, so shifts within them are not informative.

## C. Value of activity recognition (descriptive, no test of significance except where stated)
Computed from the same outputs (no additional model):
- per group: recogniser recall; **threshold agreement** (percentage of windows in which the predicted group yields
  the same threshold pair as the annotated group); false alarms per hour and episode coverage of C-oracle and
  C-predicted (episodes assigned to the annotated group at onset);
- per patient: Spearman correlation between recogniser accuracy and the false-alarm difference
  (C-predicted minus C-oracle), reported with its p-value as exploratory.

## Outputs
`C1_summary.csv` gains the task-preserving surrogate columns (defog); `C3_tests.csv` gains the C-gate rows and verdict;
new `C7_recognition_value.csv` and `C7_per_patient.csv` (prefix `FS_` for FoG-STAR).

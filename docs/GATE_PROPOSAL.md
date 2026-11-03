# Promotion Gate Review Proposal (Provisional)

This document compares supplied VAL observations with current provisional settings. It does not
select or approve thresholds. Three seeds are a small sample.

| Criterion | Saved observations (champion / seed 43) | Current provisional setting | Review note |
|---|---|---|---|
| Within-crop macro-F1 | Original supplied evaluation: champion 0.990758; seed 43 0.988771. Refreshed evaluation must be rerun with corrected matching formula before seed-44 comparison. | Candidate >= champion - 0.005 | Do not combine previous and refreshed values; recalculate after rerunning evaluator. |
| Minimum recall | 0.909091 / 0.883117 / 0.896104; Corn Cercospora, n=77; range 0.025974 | Floor 0.85 | All three observed point estimates exceed the provisional floor. |
| bg_drop, black | 0.1843 / 0.1797 / 0.1524; range 0.0319 | Candidate <= champion + 0.02 | Range exceeds the margin; fill choice affects interpretation. |
| bg_drop, mid-grey | 0.0938 / 0.0563 / 0.0997; range 0.0434 (4.34 percentage points) | Relative margin +0.02 | Margin is smaller than the observed three-run range. |
| bg_drop, per-image-mean | 0.0364 / 0.0163 / 0.0570; range 0.0407 | Relative margin +0.02 | Margin is smaller than the observed three-run range. |
| ECE | 0.001097 / 0.001387 / 0.001592; range 0.000495 | ECE <= 0.05 | Ceiling cannot reject any of these results and is far above the observed values. |

## Background criterion options

- Gate per-image-mean fill only: uses that fill's observed drop.
- Gate the mean of all three fills: summarizes them but can hide a poor individual-fill result.
- Keep black fill informational only: prevents black fill from being the sole decision input.

These are alternatives for review; no option is selected. Three seeds are a small sample.

## Tiny-class options

At `n_val = 23`, one additional error changes point recall by `1/23 = 0.043478`.

| Correct / total | Point recall | Wilson 95% lower bound (z=1.96) |
|---:|---:|---:|
| 23 / 23 | 1.0000 | 0.8569 |
| 22 / 23 | 0.9565 | 0.7901 |

Possible policies are a Wilson lower-bound floor or a point-recall floor applied only above a
minimum-support value. Neither policy nor a value is selected.

The earlier `0.8750 (70/80)` has a different denominator and does not match current VAL support
77. The refreshed reports give champion `70/77 = 0.909091`, seed 43 `68/77 = 0.883117`, and seed
44 `69/77 = 0.896104`. The original source of 70/80 is needed to explain its split/support
context; do not treat it as the same current VAL observation.

Repo search found one other unscoped crop-subset `f1_score(ct, cp, average="macro")` in
`scripts/diagnose_calibration_and_crops.py`. That call is explicitly retained as the naive
comparison beside `f1_scoped`; it is a diagnostic demonstration, not the reported per-crop metric.

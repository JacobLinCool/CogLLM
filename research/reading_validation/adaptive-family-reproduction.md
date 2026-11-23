# Reproducing material-family calibration of reading observation models

The two datasets used in the principal comparison are material partitions of the existing 24-source CogLLM archive. No additional reader data are collected. Observed ratings remain local. The archive's original receipt records source-file fingerprints and the frozen projection. Scripts are run from the repository root. The CPU experiment uses Python with NumPy, SciPy and PyTorch. The figure uses Matplotlib. The original source retrieval and L4 projection have their own documentation. [Observation archive and source fingerprints](../../docs/research/adaptive_joint/results.md) · [Historical sources and splits](borrowing-protocol.md) · [Scientific report](adaptive-family-results.md)

## Reproduce the principal comparisons

The research observation packet resides in `.cache/cogllm/adaptive-joint/inputs/observations`. It contains the source index, endpoint JSONL records, SHA-256 receipts, certified coordinates and frozen 48-component feature array. Rebuild or restore this input following the archive instructions. The model controls use the previously determined shared and private predictors. All corrected and confirmation heads are fitted anew.

```sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
python -m research.reading_validation.borrowing_global_families fit
python -m research.reading_validation.borrowing_global_families calibrate
python -m research.reading_validation.borrowing_global_families score
python -m research.reading_validation.borrowing_material_confirmation fit
python -m research.reading_validation.borrowing_material_confirmation calibrate
python -m research.reading_validation.borrowing_material_confirmation score
python -m research.reading_validation.borrowing_release_tables
python -m research.reading_validation.borrowing_family_figure
```

The fit stage seals exact role counts and globally shared parent identifiers, checks the starting forecasts, trains four CPU readouts and saves weights and observation-specific forecasts. The calibration stage sees only the allowed calibration rows and saves every sharing weight before evaluation. The evaluator verifies the design, partition, estimator and forecast receipts before scoring. Training histories record their stopping rule, selected epochs and runtime. Run reproduction in a separate checkout to preserve original receipts, which include measured elapsed time. The original material hash functions and the two earlier partitions are calculated from the source packet. [Corrected fit, calibration and evaluation](borrowing_global_families.py) · [Fourth-split material and prediction freeze](borrowing_material_confirmation.py)

## Reproduce the mechanism and earlier fixed controls

The alternative root-directory outputs keep historically examined material separate from the newly graded families. Restore the earlier shared/private forecast archives for the historical and fresh partitions or run the original training modules and historical score stage. The source-specific third comparators use independently refitted heads.

```sh
python -m research.reading_validation.borrowing_prior
python -m research.reading_validation.borrowing_replay_seals prior
python -m research.reading_validation.borrowing_prior_score
python -m research.reading_validation.borrowing_prior_diagnostics
python -m research.reading_validation.borrowing_reliability
python -m research.reading_validation.borrowing_replay_seals reliability
python -m research.reading_validation.borrowing_reliability_score
python -m research.reading_validation.borrowing_replication --train
python -m research.reading_validation.borrowing_replication_score calibrate
python -m research.reading_validation.borrowing_replication_score score
python -m research.reading_validation.borrowing_family_bootstrap fit
python -m research.reading_validation.borrowing_family_bootstrap score
python -m research.reading_validation.borrowing_family_audit
```

The replay program derives the finite-category and Gaussian divergences directly from the two predictive distributions. It checks these against the tangent residual used in calibration, then reconstructs every source loss from independently grouped sufficient statistics. A flat synthetic likelihood must return exactly 0.5. Under a convex quadratic with known mean optimum 0.7 the finite bootstrap returned 0.699882. The source-only score and nested family audit use independent resampling seeds. The figure reads numerical result files without rerunning the model. [Independent density and resampling checks](borrowing_family_audit.py) · [Synthetic checks](../results/reading-validation-family-bootstrap/numerical-test.json)

## Result files

| File within the result directory | Contents |
| --- | --- |
| `global/geometry.json` | Every endpoint's corrected split counts, overlap of original OneStop roles and quarantine counts |
| `confirmation/geometry.json` | Fourth-split eligibility, global held-family hashes and all exclusions |
| `{partition}/seed-{1729,2718}/fit.json` | Expert initialization, optimizer traces, chosen epochs, weight and prediction file fingerprints |
| `{partition}/forecast-freeze.json` | All available sharing coefficients and numerical calibration diagnostics |
| `{partition}/scores.json` | Source/component, per-seed and per-endpoint outcomes and paired uncertainty |
| `{partition}/material-statistics.npz` | Per-family weighted negative log-likelihood numerators, observed-unit denominators and anonymous family keys |
| `endpoints-calibration-and-performance.csv` | Per-source and per-endpoint coverage and scale, roles, per-seed weights, losses and calibration uncertainty, including ungraded endpoints |
| `single-expert-controls.json` | Descriptive paired differences from completely source-specific and completely shared predictions |
| `figure/adaptive-study-borrowing.csv` | Numeric data for both figure panels |

In the corrected and confirmation material-statistics files the final numerator axis indexes target MAP, equal mixture, material bootstrap, source-private expert and shared expert, in that order. The first axis indexes seeds 1729 and 2718. Distinct global family hashes represent the same article across source datasets where the original registered family ID is identical. An endpoint's held risk is its numerator summed over material divided by its denominator summed over the same material. Average risk differences inside source, then across known connected source groups. Material-bootstrap uncertainty resamples the same family for every available endpoint and retains both seeds together. [Checked release directory](../results/reading-validation-family-bootstrap) · [Numerical figure data](../results/reading-validation-family-bootstrap/figure/adaptive-study-borrowing.csv)

A new data source must define its observation scale, visible text coordinates, unit of analysis and valid predictive likelihood, then fit the corresponding observation heads. The coefficient rule can subsequently calibrate them on independent target materials. Serving the manuscript results requires these specific 48-component Qwen projections or freshly fitted heads on another backbone, plus the registered presentation geometry. A head that was measured after an entire article must receive that same exposure for the corresponding endpoint. [Joint likelihood and state definition](../union_training/observation_registry.py) · [Existing alternative-backbone results](../backbone_transfer/results.md)

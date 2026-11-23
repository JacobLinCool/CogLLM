# Fixed-weight control under retention constraints

This control uses the [retention follow-up's](retention.md) data, initialization, two seeds, optimizer update sequence, 28-epoch budget, development selection rule, and shared residual norm bound. It fixes every source multiplier at 1 within each group of microbatches. The comparison tests whether adaptive weighting improves development likelihood, behavior retention, or agreement with human effects under identical constraints. We seal the fixed and adaptive forecasts before scoring them against the existing formal evaluation outcomes.

[Original research record](../../../research/adaptive_joint/retention-control.md)

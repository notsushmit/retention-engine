# Strategy backtest

KKBox backtest figures are model estimates under assumed effects, not observed outcomes. No KKBox customer was ever given these actions, so nothing here shows that any action causes a customer to stay.

Every effect, cost and margin below is a labelled assumption (assumed effect: user choice, 2026-10-17, provisional, not estimates). Money is in price units, the unit of the KKBox price columns.

## Inputs

Customers on 2016-11-01: 169,673 (22,527 with auto-renew off, 147,146 on). No usable price: 343. Missing deadline: 317.

Paid rows dated 2016-01-01 to 2016-11-30, most common plans:

| Plan days | List price | Paid | Rows | Share |
|---|---|---|---|---|
| 30 | 149 | 149 | 973,248 | 52.0% |
| 30 | 99 | 99 | 633,843 | 33.9% |
| 30 | 129 | 129 | 103,353 | 5.5% |
| 30 | 180 | 180 | 81,111 | 4.3% |
| 30 | 150 | 150 | 27,630 | 1.5% |

The dt01 refit reproduced the rehearsal's logged score summaries exactly (3 cells). Risk comes from c01 and value from dt01, two different models: of the customers in c01's top decile of risk, 0 have a dt01 30-day lapse probability below dt01's median.

## Comparison at the headline budget (5,000), 2016-11-01

<!-- comparison table -->
*optimizer maximizes this quantity by design; the comparison is arithmetic on assumed effects.*
*effect_basis: assumed; form relative; level base; margin 0.5; cost level base; discount cost in full; value horizon 90 days; budget 5,000.*

| Strategy | Contacted | Expected retained customers | Expected retained value | Expected cost | Net value |
|---|---|---|---|---|---|
| optimizer | 4,887 | 74.7 | 10,312 | 4,999.51 | 5,313 |
| highest risk (best action) | 4,266 | 67.7 | 8,994 | 4,999.29 | 3,995 |
| highest assumed uplift x value (best action) | 4,407 | 70.7 | 9,674 | 4,999.32 | 4,675 |
| random (best action) | 4,976 (4,956 to 4,992) | 5.0 (4.5 to 5.5) | 686 (622 to 753) | 4,999.50 (4,999.02 to 4,999.97) | -4,314 (-4,378 to -4,247) |
| highest risk (reminder only) | 5,000 | 50.4 | 6,836 | 5,000.00 | 1,836 |
| no action | 0 | 0.0 | 0 | 0.00 | 0 |

Under the assumed relative base effect, the optimizer's expected retained value is 10,312 price units at an expected cost of 5,000, a net value of 5,313.

## Budget curve (base point)

| Budget | Net value | Extra net per extra unit |
|---|---|---|
| 0 | 0 |  |
| 1,000 | 2,018 | 2.018 |
| 2,500 | 3,673 | 1.103 |
| 5,000 | 5,313 | 0.656 |
| 10,000 | 6,172 | 0.172 |
| 25,000 | 6,172 | 0.000 |
| 50,000 | 6,172 | 0.000 |
| no limit | 6,172 |  |

## Sensitivity

Stability thresholds (the user's choice, not based on data): Jaccard at least 0.7 and relative regret at most 10.0%; the stricter reading (0.8, 5.0%) is shown, not judged.

| Variant | Loss | Relative regret | Jaccard | Action agreement | Stable | Stricter |
|---|---|---|---|---|---|---|
| absolute | 1,410.92 | 42.06% | 0.088 | 100.0% | no | no |
| low_effect | 4,949.26 | not defined | 0.000 | not defined | not judged | not judged |
| high_effect | 5.72 | 0.03% | 0.977 | 99.3% | yes | yes |
| margin_low | 90.83 | 7.10% | 0.740 | 99.5% | yes | no |
| margin_high | 0.00 | 0.00% | 1.000 | 100.0% | yes | yes |
| cost_low | 3,119.27 | 28.73% | 0.491 | 99.5% | no | no |
| cost_high | 1,061.27 | 69.83% | 0.412 | 94.2% | no | no |
| discount_scaled_by_margin | 458.47 | 7.85% | 0.882 | 96.2% | yes | no |
| value_365 | 445.96 | 1.33% | 0.807 | 99.7% | yes | yes |
| contact_limit | 0.14 | 0.00% | 0.999 | 100.0% | yes | yes |
| pessimistic | 3,772.37 | 3555.22% | 0.054 | 92.8% | no | no |
| optimistic | 25,408.75 | 34.19% | 0.489 | 99.3% | no | no |

Under the assumed pessimistic variant, relative regret is the loss divided by the variant's own optimum, which is small here (106.11 price units), so a loss of 3,772.37 is 3555.22% of it.

Overall verdict at the headline budget: unstable.

The plan is driven by assumptions, not data. Variants that failed: absolute, cost_low, cost_high, pessimistic, optimistic.

Assumptions ranked by loss (largest first): optimistic 25,408.75; pessimistic 3,772.37; cost_low 3,119.27; absolute 1,410.92; cost_high 1,061.27; discount_scaled_by_margin 458.47; value_365 445.96; margin_low 90.83; high_effect 5.72; contact_limit 0.14; margin_high 0.00.

Under the assumed relative low effect (zero for every action), the base plan spends 4,949.26 price units for nothing.

5 customers are chosen under every assumption except zero effect.

Discount cost rule: the discount's rank did not change in this run, but the choice of cost rule remains an assumption the result rests on.

Recommendation: a randomized holdout test of the action behind the largest loss before any use of this plan.

Expectations registered before the run:

- the low_effect variant gives an empty plan
- the absolute form is likely the least stable (a flat effect favours high-value customers whatever their risk)
- the margin and cost variants mostly change spend more than who is targeted

## Secondary snapshot

2016-10-01 holds the half of customers used to fit the Platt calibration, so its probabilities are partly in-sample.

<!-- comparison table -->
*optimizer maximizes this quantity by design; the comparison is arithmetic on assumed effects.*
*effect_basis: assumed; form relative; level base; margin 0.5; cost level base; discount cost in full; value horizon 90 days; budget 5,000.*

| Strategy | Contacted | Expected retained customers | Expected retained value | Expected cost | Net value |
|---|---|---|---|---|---|
| optimizer | 4,922 | 77.5 | 10,744 | 4,999.22 | 5,745 |
| highest risk (best action) | 4,299 | 70.9 | 9,579 | 4,999.48 | 4,579 |
| highest assumed uplift x value (best action) | 4,561 | 73.8 | 10,174 | 4,999.56 | 5,174 |
| random (best action) | 4,974 (4,953 to 4,990) | 6.2 (5.6 to 6.7) | 856 (787 to 927) | 4,999.49 (4,999.02 to 4,999.97) | -4,144 (-4,213 to -4,073) |
| highest risk (reminder only) | 5,000 | 53.5 | 7,291 | 5,000.00 | 2,291 |
| no action | 0 | 0.0 | 0 | 0.00 | 0 |

## Observed churn among targeted customers

*says nothing about whether any action works.*

| Snapshot | Strategy | Targeted | Share lapsed within 30 days |
|---|---|---|---|
| 2016-11-01 | optimizer | 4,887 | 38.5% |
| 2016-11-01 | highest risk (best action) | 4,266 | 46.8% |
| 2016-11-01 | highest assumed uplift x value (best action) | 4,407 | 38.3% |
| 2016-11-01 | random (best action) | 4,976 | 3.9% |
| 2016-11-01 | highest risk (reminder only) | 5,000 | 44.5% |
| 2016-10-01 | optimizer | 4,922 | 38.8% |
| 2016-10-01 | highest risk (best action) | 4,299 | 46.2% |
| 2016-10-01 | highest assumed uplift x value (best action) | 4,561 | 37.5% |
| 2016-10-01 | random (best action) | 4,974 | 4.5% |
| 2016-10-01 | highest risk (reminder only) | 5,000 | 44.0% |

## Solver checks

Cells checked by HiGHS: 112; proven optimal within 120 s: 79; not proven: 33, of which with no feasible HiGHS solution: 21. Budget or limit breaches: 0; repairs: 0.

Largest HiGHS minus greedy on proven cells without a contact limit: 1.194359. On proven contact-limit cells, against greedy cut to the limit: 35.904906.

The exact check is limited: in 33 of 112 cells (29.5%) HiGHS did not prove the optimum within 120 s. Those cells keep greedy's plan, checked only against its LP bound, except the contact-limit cells, where the plan is HiGHS's best feasible solution only if it is worth at least greedy's (greedy cut to the limit). Unproven contact-limit cells: 1: budget 2,500 (2016-11-01), plan from HiGHS.

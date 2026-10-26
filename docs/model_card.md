# Model card

A short card for the models in this repository. Every figure is copied from a generated report or a config in this repository (the report is named beside it), and the card inherits every caveat of those reports.

> The service replays 2016 KKBox data; there is no live source.
>
> Customers expiring more than 30 days ahead (22% of active customers on 2016-11-01) get no score.
>
> Plans exist for two dates only (2016-10-01 and 2016-11-01) and were built from the 30-day probability before the ordering fix.
>
> KKBox backtest figures are model estimates under assumed effects, not observed outcomes.

## What the models are

- **Classifier (c01):** one LightGBM model per horizon gives the chance that a paying customer lapses within 7, 30 and 59 days of the snapshot date. Platt calibration and an ordering fix make the three probabilities consistent. Every 59-day probability is a demonstration: no validation fold chose that model's settings.
- **Survival model (dt01):** a discrete-time model that gives the shape of the curve over the following days; the registered goals for it at 7 days were not met (below).
- **Baseline:** a banded logistic regression, the strongest simple baseline.
- **Reasons:** per-customer contributions of the classifier, written from templates. They describe what moved the model's number, not why a customer behaves as they do.
- **Budget optimizer:** chooses customers and actions for a budget under assumed effects.

## Intended use

Ranking paying customers who expire within 30 days by their chance of lapsing, reading what the model sees in them, and comparing budget plans under stated assumptions. It is not evidence of what an action would do for a customer. The service and the dashboard replay recorded 2016 data at one store date (2016-11-01); nothing in this repository is hosted.

## Data

- The KKBox churn prediction challenge (WSDM Cup 2018), a 20% customer sample chosen by a hash of the customer ID. The files are competition data under the Kaggle competition rules: they are not in this repository and are not redistributed. The KKBox data terms are not part of this repository.
- A customer lapses when their paid membership ends and no new subscription follows within 30 days (day 30 included); the lapse date is that expiry (see the [churn definition](churn_definition.md)).
- The classifier is fit on the snapshots 2016-08 and 2016-09 and calibrated on part of 2016-10; the test months 2017-01 and 2017-02 were scored once.

## Results

From the [model comparison report](../reports/model_comparison.md), 2017-01, paying customers expiring within 30 days:

| Model | 7-day AUC | 30-day AUC |
| --- | --- | --- |
| LightGBM classifier c01 | 0.9393 [0.9341, 0.9445] | 0.8429 [0.8374, 0.8484] |
| Survival model dt01 | 0.9054 [0.8971, 0.9136] | 0.8414 [0.8359, 0.8469] |
| Banded logistic baseline | 0.8977 [0.8902, 0.9054] | 0.7687 [0.7623, 0.7750] |

- The survival model was worse than the classifier at 7 days: the AUC difference is -0.0339 in 2017-01 and -0.0527 in 2017-02, so its registered goals at 7 days were not met.
- Scored cell for 30 days in 2017-01: 150,672 rows and 7,982 lapses.
- Replayed store against batch scoring: 169,673 scored customers, largest gap 0.0, at one store date ([serving parity](../reports/serving_parity.md)). Over HTTP, 996 of 1,000 answers matched as written and 1,000 of 1,000 only under amendment A10.
- Under the assumed relative base effect, the budget plan at 5,000 price units on 2016-11-01 has an optimizer net value of 5,313 against 3,995 for contacting the highest risk ([strategy backtest](../reports/strategy_backtest.md)).

## Limits

- One test month decides the 30-day result. The intervals cover which customers were in the month, not month-to-month change.
- December 2016 is never explained: renewals are missing on 2016-12-05 and 2017-02-13, and the data cannot say whether they are missing from the data or real billing failures. The 7-day lapse rate of 2016-11 has an excess in a small group of customers whose last row was free of charge; it is not explained.
- Only paying customers expiring within 30 days are scored. Customers outside that population get no score.
- A fixed set of profile attributes is never shown as a reason, a lever, a segment or a filter.
- Plans exist for two dates, and nothing is re-planned. The latency figure in the [README](../README.md) was measured before the 1.5 GiB memory limit and before the dashboard endpoints.

## Causal caveats

No KKBox discount effect was identified: the treatment-effect method was checked on two public randomized datasets, and on KKBox the groups were too unbalanced to estimate anything ([causal assumptions](causal_assumptions.md)). Every action effect in the plans is therefore an assumption, and the plan is unstable under 5 of 11 variants of those assumptions.

## Drift and retraining

Drift is demonstrated on synthetic data only, never on KKBox. On synthetic data the dataset-level alert rule was NOT MET (shifted share 0.2979, below 0.5) and the margin rule was met (0.2553 against 0.20). The real drift replays on the 20% sample are NOT RUN; the retraining flow and the one real retraining run are NOT DONE. The synthetic data are generated by hand-set rules; they are not KKBox data and say nothing about KKBox customers. Why the two 365-day count columns drift without the shift on synthetic data is not tested. Reports under `configs/ops.yaml`, amendment A16.

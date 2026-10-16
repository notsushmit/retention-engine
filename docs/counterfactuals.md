# Counterfactuals of the churn model

A counterfactual here is the churn model's score under a changed input. It answers "what would the model score if this input were recorded differently?", not "what would happen if we changed it?". **Every counterfactual is the model's score under a changed input and never an effect.** Only an uplift estimate can support a claim about what an action does, and no KKBox uplift estimate exists (see `docs/causal_assumptions.md`).

Everything below was registered in `configs/uplift.yaml` and `configs/counterfactual_phrases.yaml` before the run. Every number comes from the one logged run (`python -m retention.explain counterfactuals`).

## Model and rows

- **Model:** the frozen 30-day churn model, loaded from its saved file with a checksum check and never refit. Its scores are calibrated probabilities.
- **Rows:** the customers already explained by the reasons step: those expiring within 30 days at 2016-10-01, 2016-11-01, 2017-01-01 and 2017-02-01, 645,753 rows in all.
- **Reads:** only features. The run read no outcome; its audited label reads asked only which rows are in the group.

## The one lever

- **Auto-renew,** from off to on only. It is never switched off.
- **Support:** auto-renew is offered for a row only if comparable customers in the model's training rows (303,072 customers expiring within 30 days at 2016-08-01 and 2016-09-01) include at least 10 with auto-renew on. Comparable means the same plan length, list price and payment method, and similar listening days and days until expiry. 226 such groups have that support.
- **Minimum:** a change is shown only if it lowers the calibrated 30-day churn score by at least 1 percentage point. That minimum is the user's own registered choice and was not validated.

## Deviations from the project plan

- **No dice-ml.** A small constrained search over the one lever needs no new dependency.
- **No plan-length lever.** Longer plans at their usual price are rare in the model's training rows (counts in `configs/uplift.yaml`), and plan length is a choice the customer makes.
- **No usage lever.** The listening features are tightly linked (days, seconds, plays, ratios, trends) and cannot be changed consistently one at a time.
- **Never levers:** age, gender, registration channel, tenure and city.
- **Not treated as levers:**
  - price and discount, because the KKBox discount effect could not be estimated;
  - payment method;
  - anything about the past or the calendar.

## Coverage

Rows per snapshot, by what each row got:

| Snapshot | Counterfactual shown | No: below the minimum | No: auto-renew on would raise the score | No: no comparable support | Auto-renew already on |
|---|---|---|---|---|---|
| 2016-10-01 | 4,081 | 12 | 16 | 17,467 | 123,420 |
| 2016-11-01 | 4,116 | 8 | 19 | 18,384 | 147,146 |
| 2017-01-01 | 4,708 | 8 | 20 | 18,027 | 127,909 |
| 2017-02-01 | 4,850 | 3 | 13 | 17,529 | 158,017 |

- **2.7% of the explained rows carry a counterfactual:** 17,755 of 645,753.
- An earlier planning figure of about 3.4% counted auto-renew-off rows that had comparable auto-renew-on customers in their own snapshot. That was an upper bound under a looser support rule, not coverage.
- Most rows get none for one of two reasons:
  - auto-renew is already on, so the lever does not apply;
  - no comparable training customers have auto-renew on. Auto-renew goes almost entirely with the payment method, so for most customers with it off, "auto-renew on" is a combination the data do not contain.

## The Done-when check

**Passed.** No counterfactual changes any feature outside the lever. The run checks this on every snapshot and stops on a breach: on all four, only auto-renew changed, and it was never switched off.

## Wording

The lines come from `configs/counterfactual_phrases.yaml`:

> With auto-renew recorded as on, the model's {horizon}-day churn score would be {new} instead of {old}.
> This is the model's score under a changed input, not the effect of any action; customers with auto-renew on may simply be customers who were staying anyway.

There are also lines for "would be higher", for no comparable support, and for a change below the minimum.

A structure test enforces the wording rules:
- no causal, advice, lever or action-claim words;
- digits only inside the fact fields;
- nothing about age, gender, registration channel, tenure or city.

## Run

One run: 5.1 seconds, peak memory 1.61 GB, no swap used.

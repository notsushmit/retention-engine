# Churn archetypes and reasons

The goal was named churn types: a small, stable set of groups of customers whose churn risk is pushed by the same things. The attempt failed by its own registered rule. No archetype set was found, and none is stored. This page records what was tried, the result, what is still defined by rule, and the notes a reader of the per-customer reasons needs.

All figures come from the 20% customer sample.

## What was tried

Everything below was written in `configs/archetypes.yaml` and committed before any clustering code ran. The pass marks are this project's own choices, made in advance; they are not taken from a paper.

- **Rows.** Customers whose coverage expires within 30 days of the snapshot date, in the `paid` or `discount100_only` segment, at four monthly snapshots: 2016-10-01, 2016-11-01, 2017-01-01 and 2017-02-01. That is 645,753 rows. 2016-12-01 is not explained, because no outcome window at 7 or 30 days is usable for it.
- **What each row looks like.** Eleven numbers per row: how hard each reason group pushes the 30-day churn score up or down (its SHAP contribution, on the calibrated log-odds scale). Nothing else goes in: no raw facts and no rescaling. The registration channel is left out, and a missing tenure counts as 0.
- **Set aside by rule first.** Rows whose last plan was free (trial, promotion or 100% discount) form their own group, described below. That is 3,687 of the 645,753 rows.
- **Method.** HDBSCAN from scikit-learn (euclidean distance, `min_samples` 25), with no dimension reduction first. HDBSCAN finds dense groups and labels points that fit no group as "unassigned". It was fitted on 40,000 rows: 20,000 each from 2016-10 and 2016-11. Every other row takes the majority label of its 10 nearest fitted rows.
- **Settings tried.** The smallest allowed group size (`min_cluster_size`) was set to 200, 400 or 800 rows: 0.5%, 1% and 2% of the fit sample. No other setting was allowed.
- **How a setting is chosen, with no outcome used.** A setting qualifies if it finds 3 to 12 groups and leaves at most 20% of a separate 20,000-row evaluation set unassigned. Among the settings that qualify, the one whose groups come back most reliably wins. Reliability is the median adjusted Rand index (ARI) over 20 refits on fresh samples. ARI measures agreement between two groupings of the same rows: 1 means identical, about 0 means no better than chance. Ties go to the larger size. If no setting qualifies, that is reported as the result and nothing else is tried.
- **What would have followed.** These steps needed a chosen setting:
  - a check of whether the groups mostly restate how many days are left until expiry;
  - seven stability tests: the same groups in refits, in each 2016 month on its own, with the same push pattern each month, with steady shares each month, found again in an independent 2017 fit, at most 20% unassigned each month, and 3 to 12 groups in all.

## Result

No setting qualifies. Every setting finds more than 12 groups.

| Smallest group size | Groups found | Unassigned on the evaluation set | Median ARI over 20 refits (range) | Qualifies |
| --- | --- | --- | --- | --- |
| 200 | 34 | 15.66% (3,132 of 20,000) | 0.8159 (0.7486 to 0.9115) | no: more than 12 groups |
| 400 | 18 | 16.77% (3,354 of 20,000) | 0.7910 (0.7585 to 0.9144) | no: more than 12 groups |
| 800 | 14 | 14.70% (2,940 of 20,000) | 0.9012 (0.8289 to 0.9273) | no: more than 12 groups |

Unassigned shares among the other rows. These are reported only; the evaluation set alone decides.

| Smallest group size | Fit-sample rows (of 40,000) | Other non-rule rows (of 602,066) | All rows, rule group counted as assigned (of 645,753) |
| --- | --- | --- | --- |
| 200 | 18.20% (7,282) | 16.77% (100,995) | 16.77% (108,277) |
| 400 | 19.06% (7,626) | 17.50% (105,386) | 17.50% (113,012) |
| 800 | 17.17% (6,867) | 15.16% (91,264) | 15.20% (98,131) |

Every setting stays under the 20% unassigned limit. All three fail on the number of groups alone. The full grid was run twice, and both runs gave identical results and identical labels for every row. Each run took about 7 minutes and used at most 2.1 GB of memory.

The expiry check and the seven stability tests were not run, because both need a chosen setting. No archetype table and no monthly share series were written.

## What it means

The refits agree with each other well (median ARI 0.79 to 0.90). So the space of reason pushes has stable fine structure. But at the registered settings that structure did not reduce to 3 to 12 groups: even with groups of at least 800 rows, 14 were found. Why is not known. Once the result was in, the groups' contents were deliberately not inspected, so nothing here describes them.

This does not show that customers have no churn types. It shows only that this registered method, at these settings, found no small stable set of them. No other setting or method was tried after the result.

For this phase, the per-customer reasons are complete and pass their automatic checks. "Archetypes that hold up across two snapshot months" was not achieved.

A rule-based set of named segments could be tried later if a later step needs one. It would be registered separately before it runs and shown as a second attempt, with this failure kept beside it.

## What is defined by rule

**Customers whose last plan was free.** These are customers whose last transaction before the snapshot was a trial, a promotion or a 100% discount: 3,687 of the 645,753 explained rows. This group is defined by rule, not found by clustering, and it was set aside before clustering. It was kept apart because it is small but carries a large part of the short-term churn. In November 2016, this group was about 1% of rows and about 42% of 7-day lapses. That is one month's outcome figure, not a measure across months. No share series or profile was produced for it in this phase.

## Notes for readers of the reasons

Reasons are stored for every explained row at 7 and 30 days: 1,291,506 rows in all (645,753 per horizon). A reason says what the model is picking up on, not what caused anything. Counts below pool the four explained snapshots unless a month is named.

- **Which model is explained.** The reasons explain the project's LightGBM churn classifier at 7 and 30 days. It was refit for this purpose, exactly as the evaluation fits it. The refit was checked field by field against the summaries logged by the 2016 rehearsal run: mean, nine deciles, minimum, maximum and number of distinct scores, 39 of 39 fields equal over three checks. The final test run's own result file was never opened. So the check ties the refit to the rehearsal's model, and to the test run's model only through the same code, the same inputs and a deterministic fit.
- **"Typical customer", not the average probability.** The position line ("higher than", "lower than" or "close to a typical customer in the training months") compares with the model's base value. That is the average raw log-odds over the 303,072 training rows (2016-08 and 2016-09). Averaging log-odds lands near a typical customer's score, not near the average probability: about 0.20% at 7 days and 1.88% at 30 days as probabilities. For comparison, in November 2016 (169,673 rows) the median 7-day score was 0.177% while the mean was 1.897%.
- **The position is the net, not the strongest reason.** The reasons are separate pushes, and the position line is what they add up to. So the position can be weaker than the strongest reason, point the other way, or read "close" next to a strong reason. When the position is weaker than the strongest listed reason and something visibly pulls the other way, a last line says so: "These pull in different directions, so the overall risk...". That happens on 219,437 of 645,753 rows at 7 days and 169,492 of 645,753 at 30 days.
- **The strength label is not a probability.** "Slight", "moderate" and "strong" mean that a push multiplies or divides the odds by at least 1.2, 1.5 or 2, compared with a typical training customer. A strong label can sit next to a low probability. For example, a 0.7% chance of lapsing within 7 days is still several times a typical customer's.
- **The 7-day and 30-day probabilities can disagree.** Each displayed probability is calibrated (Platt scaling) for its own time window by a separate model. Until the fix that puts the three horizons in order, a customer's 30-day probability can come out below their 7-day one. That fix needs the 59-day model and comes with the scoring service.
- **Very small and very large probabilities.** A probability below 0.1% shows as "under 0.1%", and one above 99.9% as "over 99.9%". Exactly 0.1% shows as "0.1%". At 7 days, 165,519 of the 645,753 rows show "under 0.1%". No 30-day row is below 0.1%, and no row at either horizon is above 99.9%.
- **A missing listening-trend ratio.** The ratio of the last 7 days to the three weeks before is missing when the customer has fewer than 3 listening days in those three weeks, not only when there are none. The fact then reads "too little listening in the three weeks before to compare with". A missing value can still push risk, because the model treats missingness as information.
- **"About N minutes a day"** is the listening time of the last 28 days divided by all 28 calendar days, not by the days the customer listened. A customer who listened for 278.6 minutes on a single day reads "about 10 minutes a day".
- **Billing timing has two readings.** At 7 days it mostly means coverage ending soon: where it pushes risk up, 99.3% of rows expire within 7 days (174,134 of 175,406). At 30 days it mostly means a long gap since the last payment: where it pushes risk up, 88.7% of rows last paid more than 31 days ago (21,088 of the 23,781 with a known last payment). That is where customers whose last plan was free show up.
- **"How they pay" at 7 days.** Where it pushes risk up, the customer has auto-renew on in 92.1% of rows (324,810 of 352,846). Those customers are mostly far from expiry, not close to it: 2.4% expire within 7 days (7,761 of 324,810), with a median of 19 days to expiry. This describes how the model splits credit between related inputs. It does not claim that auto-renew raises risk.
- **Amounts paid** are shown as recorded. The data does not state a currency.
- **Group pushes and Shapley values.** A reason group's push is the sum of its inputs' contributions. That sum can differ from the Shapley value of the group treated as a single input, but only through interactions with inputs outside the group.
- **Two expiry columns, one reason.** `days_until_expiry` (coverage) and `days_until_paid_expiry` (paid coverage) are equal on 98.54% of the 303,072 training rows (298,661) and on 98.94% of the 301,853 rows that have a paid expiry. The other 1,219 training rows were never paid and have none. Both columns sit in the billing-timing group, so they always appear together as one reason.

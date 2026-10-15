# Churn change points and alerts

The goal was to spot automatically when the churn rate shifts and which segment is behind it. The pieces exist and ran, and the result is exploratory: the detector's mechanics were tested on synthetic series, but its false-alarm rate on this real series was not validated. This page records the design as it was registered, what each check showed, and what the alerts can and cannot say.

All figures come from the 20% customer sample. The registered settings are in `configs/changepoints.yaml`; they were committed before the synthetic study ran and before the detector read any real series. A change after a result was seen is recorded as an amendment, not as a new setting.

## The series

- **Event.** A customer with an effective expiry date in a week. The week is the Monday-to-Sunday week of that expiry. Each expiry date appears once, taken from the snapshot of its own month.
- **Churn.** No new subscription within 30 days after the expiry, inclusive (the coverage-based label).
- **Series.** `paid` is the headline. `discount100_only`, `trial_only` and `promo_only` are separate series and are never pooled into it.
- **Range.** Weeks starting 2016-02-15 to 2016-12-19, which is 45 weeks. Earlier weeks carry a left-truncation flag; the 2015 weeks are not part of detection. A week counts only if its end plus 30 days is on or before 2017-03-31.
- **The series ends at 2016-12-31.** Outcomes of 2017 snapshots are locked, and no new read path to them was added. So 2017 expiries, including the February 2017 gap day and the Lunar New Year period, are not in the change-point series. The daily indicator below uses payment rows only, so it does cover them.
- **Minimum size.** A weekly cell needs at least 1,838 events, the smallest count at which a rate near 5% has a Wilson 95% interval narrower than +/-1 percentage point (200 events gives +/-3.1). Paid weeks are far above it: 45 of 45 weeks, median 35,193, minimum 22,817, maximum 56,002. The other segments are far below it: `discount100_only` has a median of 223 events a week (maximum 782), `trial_only` a median of 3 (maximum 7,912) and `promo_only` a median of 8 (maximum 43). They are therefore reported by month or for the whole period, and are never in a detection run.
- **Segment and month.** The segment is the one on the first of the month, used for the whole month's expiries.

### Month-end pattern

Weekly counts swing around month ends (22,817 events in the week of 2016-10-24 and 56,002 in the week after). Before any detection, the paid weekly rate of the 15 weeks that contain one of the last four days of a month was compared with the other 30 weeks: means 0.0672 and 0.0671, Welch t = 0.013. No month-end pattern was found, so detection ran on the unadjusted series. With 45 weeks the test is weak, so this means "not detected".

## Daily batch-size indicator

The monthly data-quality screen missed single missing days. A daily indicator counts non-cancel auto-renew payment rows per transaction date and compares each unit with the median of the same unit over the preceding three months (at least two present). Days 2 to 27 are judged one by one. The last four days of a month are judged together as one block by position from the end, so a batch that moves from the 30th to the 31st is not flagged. A unit below 0.60 of its baseline is a gap. No weekday effect was found in the clean months (the largest weekday median differs from 1 by 0.022 against a day-to-day noise of 0.078), so no weekday baseline is used.

- **Clean months, March to November 2016.** 237 units judged, 0 flagged (the registered limit was 1). The lowest ratio was 0.769.
- **Known days, scored after the config was committed.**

| Day or block | Count | Baseline | Ratio | Flagged |
| --- | --- | --- | --- | --- |
| 2016-12-05 | 2,005 | 5,066 | 0.396 | yes |
| 2017-02-13 | 2,100 | 4,834 | 0.434 | yes |
| 2016-12-30 (block 12-28 to 12-31) | 40,738 | 44,588 | 0.914 | no |

The two days either side of each day had ratios between 1.026 and 1.067, none flagged. The February 2017 month-end block was 32% above its baseline; the rule is one-sided, so this is an observation only. All registered pass marks were met and nothing was tuned.

**Limits.** It catches large gaps, not moderate ones: in the clean months the median ratio is 1.041, so the baseline sits slightly low and a drop to between 60% and 100% of normal is not flagged. A shortfall inside a month-end block that is offset by an overflow in the same block is not flagged. A week of expiries is marked `renewal_gap` if a gap day falls from the week's start to its end plus 30 days.

## The detector

- **Method.** PELT from the `ruptures` package with an L2 cost (it detects shifts in the mean) and a minimum segment of 3 weeks. The penalty is `c * sigma2 * log(T)`, where `sigma2` is the binomial variance of a weekly rate and `T` the number of points. Only `c` was chosen, on synthetic series only, from the grid 1, 2, 3 and 5.
- **Tolerance.** A detection counts if a returned change point is within +/-1 week of the true shift. The grain is weekly, so a tolerance of a few days cannot be resolved.

### Synthetic study

Series of 40 weeks, noise taken from the real weekly event counts (counts only), 500 replicates per cell, three base rates (5%, 10%, 20%), seed 20161101. The noise was binomial scaled by 1.0, 1.5 and 2.0; the marks were judged at 1.5 times binomial noise. It ran in 408 seconds with a peak of 0.99 GB.

| c | Mean false alarms per stable stretch (5%, 10%, 20%) | Share of stretches with any (limit 0.10) | Marks |
| --- | --- | --- | --- |
| 1 | 4.26, 4.24, 4.07 | 0.99, 0.97, 0.98 | not met |
| 2 | 1.55, 1.53, 1.43 | 0.67, 0.65, 0.67 | not met |
| 3 | 0.56, 0.55, 0.47 | 0.32, 0.32, 0.31 | not met |
| 5 | 0.052, 0.068, 0.076 | 0.044, 0.054, 0.064 | met |

- **Chosen c = 5.** It is the last value in the grid, so there is no headroom above it.
- **Detection at c = 5 and 1.5 times noise.** Shifts of 3 and 5 points were found within tolerance in every replicate. 1-point shifts were found in 100%, 99.8% and 97.6% of replicates (5%, 10%, 20%), and two shifts in one series were both found every time. At pure binomial noise there were no false alarms in 500 stretches at any base rate.
- **At 2 times noise (reported, not a mark).** False alarms rise to 0.64, 0.63 and 0.59 per stretch, with 34% to 38% of stretches having at least one, so the false-alarm marks would not be met. The detector breaks on false alarms between 1.5 and 2 times noise; detection stays strong.
- **Spikes.** A spike of 3 or 4 times the base rate lasting 1 or 4 weeks produced about two change points, one at each edge. With a minimum segment of 3 weeks, a 1-week spike is widened to a 3-week segment, so its two change points sit one week either side. A spike is not ignored, which is why alerts around spikes carry a data-quality flag.
- **What this validates.** Real quiet-week variation in the paid series is about 0.4 to 0.55 points, roughly four to five times the binomial noise of about 0.11 points. The study therefore validates the detector's mechanics, not its false-alarm rate on this series. Real-series alerts are exploratory and are labelled "validated at 1.5x binomial noise".

## Alerts on the real series (exploratory)

The detector ran once on the paid weekly series, with every setting registered. It returned 10 change points, and 28 of the 45 weeks carry a data-quality flag. Each change point has an alert in `data/processed/alerts.json`.

| Week | Rate before to after | Size (points) | Class | Flags |
| --- | --- | --- | --- | --- |
| 2016-03-07 | 11.9% to 18.1% | +6.1 | shift | unreliable month (2016-03), uncertain (2016-02) |
| 2016-03-28 | 18.1% to 7.3% | -10.7 | shift | unreliable month (2016-03), uncertain (2016-04) |
| 2016-04-18 | 7.3% to 7.9% | +0.5 | small | unreliable month (2016-05), uncertain (2016-04) |
| 2016-05-09 | 7.9% to 5.4% | -2.5 | shift | unreliable month (2016-05) |
| 2016-07-04 | 5.4% to 4.6% | -0.8 | small | uncertain (2016-06) |
| 2016-08-22 | 4.6% to 5.2% | +0.6 | small | none |
| 2016-09-26 | 5.2% to 4.5% | -0.7 | small | known period (September 2016 level shift) |
| 2016-10-24 | 4.5% to 5.5% | +1.0 | shift | renewal window of the gap on 2016-12-05 |
| 2016-11-14 | 5.5% to 4.6% | -1.0 | small | renewal window of the gap on 2016-12-05, known period (November 2016 promotion wave) |
| 2016-12-05 | 4.6% to 7.6% | +3.1 | shift | data gap 2016-12-05, uncertain (2016-12), known period |

- **The February to March 2016 excursion** (weekly rates of 15% to 23% over four weeks) is in a month the data-quality screen marks as unreliable. It is reported with that flag and not as a change in customer behaviour; no missing days were confirmed there.
- **The week of 2016-12-05** (rate 12.5%) coincides with the one confirmed gap day: auto-renew payment rows that day were about 40% of normal (ratio 0.396). It is reported as a data gap.
- **The September 2016 shift** does not show as a clear change in the paid rate: its alert is small (-0.7 points) and sits in a known period.
- **Alert sentences** follow the rules of the per-customer reasons: no causal words, no advice or lever words, digits only in fact fields. A flagged alert never reads as churn worsening.
- **Small changes.** A change below 1.0 point is reported as "detected, small" and no segments are ranked.

### What the quiet-week rule did not deliver

The rule that was registered measures the quiet-week standard deviation over weeks that are not flagged and that sit in detector segments of at least 8 weeks. On this series it was not computable: the only segment of 8 weeks or more (2016-05-09 to 2016-06-27) is entirely flagged. Two consequences follow, and the value was not back-filled from other estimates:

- The 1.0-point threshold for "small" was not validated against a measured noise level on this series.
- `series_returned_to_level` is not computable and is null for every alert.

### Drill-down

For each non-small alert, segments are compared before and after by plan length, payment method, auto-renew, city, registration channel and tenure bucket. A segment needs at least 1,838 events in both periods; smaller ones are listed as too small and not ranked. The change is split into a rate part and a mix part, and the parts add up to the overall change in every dimension.

- **It ranks large groups first.** Rate change times share rewards size, so the top segments are mostly the largest groups (plan length 8 to 30 days holds 94% to 98% of events; auto-renew on, 81% to 88%). It does not show where a change is concentrated.
- **A first-run artefact was fixed.** In the first run, events of customers with no member profile (about 16%) were dropped from the city and registration channel dimensions, which had made the registration-channel ranking partly an artefact. They are now labelled "missing" and counted, and all six dimensions sum to the same total. The change points and rates did not move.
- Registration channel is a descriptive segment only, never a lever. Age, gender and the profile group are not used.

### Code defects found on the first real run

Three defects were found and fixed, with no registered value changed: a missing band was reported as "not returned" instead of "not computable"; missing city and channel values were dropped; and there was no guard for a window shorter than the minimum, which would have printed "nan%". The third was a missing guard, not a registered value. The guard was not triggered on this series.

## What the detector cannot show

- **Cause.** An alert says that the rate moved and where, never why.
- **Small segments.** Segments under 1,838 events in either period are not ranked; the small series (`discount100_only`, `trial_only`, `promo_only`) are not in detection.
- **Timing within a week.** The grain is weekly, and the tolerance is one week.
- **A one-week spike at the end of the series.** The minimum segment of 3 weeks merges it into the final segment (the week of 2016-12-05), so no change point isolates it.
- **The first and last weeks.** A change within about 3 weeks of either end of the series cannot be seen.
- **2017.** The series stops at 2016-12-31.

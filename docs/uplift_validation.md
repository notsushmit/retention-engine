# Uplift validation on randomized data

This page reports the one registered validation run of the uplift learners on two public datasets where chance decided who received the action. The aim was to check whether the learners rank customers by the effect of an action better than chance would. Everything was registered in `configs/uplift.yaml` before either dataset was used for a model. Every number below comes from the logged run (`python -m retention.uplift validate`, one run).

## Data

- **Hillstrom:** an email test with 64,000 customers, randomly split into a men's email, a women's email and no email (Kevin Hillstrom, MineThatData E-Mail Analytics and Data Mining Challenge, 2008). The two cells, men's against none and women's against none, share the no-email customers, so they are not independent.
- **Criteo:** an advertising test with 13,979,592 users, 85% treated (Diemert, Betlei, Renaudin and Amini, "A Large Scale Benchmark for Uplift Modeling", AdKDD 2018 workshop; version 2.1, CC BY-NC-SA 4.0). The `exposure` column records something after assignment, so it was never a feature; the analysis uses the assigned treatment.
- Neither dataset is stored in this repository.

## Method in brief

- Each dataset was split once, half for training and half for evaluation, with a fixed seed, keeping the arms in proportion. All choices were made on the training half; the evaluation half was used only to score.
- Two learners were fitted with LightGBM. The X-learner was registered in advance as the one judged; the T-learner is reported beside it. Model settings were chosen from a small registered grid by cross-validation on the training half.
- **Qini curve:** customers are ranked by predicted uplift. At each of 100 cut points, it takes the extra outcomes among the treated in the top group, after scaling the control group's outcomes to the same size.
- **Qini area (A):** how far that curve lies above the straight line a random ranking would give, averaged over the cut points. A random ranking scores 0 on average.
- **Intervals** are 95% Poisson-bootstrap intervals with 1,000 resamples. They cover sampling variation in the evaluation half, not variation from training the models.

## Pass marks, registered in advance

A primary cell is met only if both marks pass:

- **(a)** the lower end of the X-learner's interval for A is above 0;
- **(b)** A is at least 10% of the area under the random line, i.e. A ≥ 0.10 × Q(1) / 2, where Q(1) is the whole evaluation group's extra outcome per customer.

Mark (b) is judged only when the naive treated-minus-control difference is clearly above zero. A cell that passes (a) without that is reported as "passed on (a) alone" and counts as not met.

The first Done-when criterion requires all three primary cells (visit in both Hillstrom cells, and in Criteo) to be met.

## Result: the first Done-when criterion is not met

The men's Hillstrom cell fails both marks; the other two primary cells pass. This is reported as found. There was no retuning, no change of the judged learner, and no dropped cell.

### X-learner pass marks

| Cell | Evaluation customers | X-learner A, 95% interval | A | Q(1) | Mark (b) needs A ≥ | (a) | (b) | Verdict |
|---|---|---|---|---|---|---|---|---|
| Hillstrom men's, visit (primary) | 21,307 | -0.00072 to 0.00229 | 0.00085 | 0.03801 | 0.00190 | not met | not met | **not met** |
| Hillstrom women's, visit (primary) | 21,346 | 0.00235 to 0.00513 | 0.00372 | 0.02355 | 0.00118 | met | met | met |
| Criteo, visit (primary) | 6,989,796 | 0.00296 to 0.00324 | 0.00310 | 0.00879 | 0.00044 | met | met | met |
| Hillstrom men's, conversion | 21,307 | -0.00050 to 0.00027 | -0.00012 | 0.00347 | 0.00017 | not met | not met | not met |
| Hillstrom women's, conversion | 21,346 | -0.00013 to 0.00059 | 0.00024 | 0.00088 | not judged | not met | not judged | not met |
| Criteo, conversion | 6,989,796 | 0.00035 to 0.00042 | 0.00039 | 0.00098 | 0.00005 | met | met | met |

The conversion cells are secondary: reported, not judged for the Done-when.

### The other rankings (Qini area: 95% interval, then estimate)

| Cell | T-learner | Outcome-probability ranking | Random ranking |
|---|---|---|---|
| Hillstrom men's, visit | -0.00108 to 0.00184; 0.00039 | 0.00096 to 0.00358; 0.00227 | -0.00035 to 0.00232; 0.00091 |
| Hillstrom women's, visit | 0.00188 to 0.00463; 0.00326 | 0.00084 to 0.00322; 0.00202 | -0.00168 to 0.00091; -0.00035 |
| Criteo, visit | 0.00281 to 0.00310; 0.00296 | 0.00288 to 0.00316; 0.00302 | -0.00012 to 0.00007; -0.00003 |
| Hillstrom men's, conversion | -0.00063 to 0.00015; -0.00025 | -0.00023 to 0.00057; 0.00016 | -0.00015 to 0.00061; 0.00022 |
| Hillstrom women's, conversion | -0.00007 to 0.00062; 0.00026 | -0.00035 to 0.00027; -0.00004 | -0.00059 to 0.00004; -0.00026 |
| Criteo, conversion | 0.00028 to 0.00036; 0.00032 | 0.00036 to 0.00044; 0.00040 | -0.00002 to 0.00002; -0.00000 |

The outcome-probability ranking orders customers by how likely they are to visit or convert, ignoring the action: the equivalent of targeting by risk.

### Uplift added little over targeting by risk

The paired difference between the X-learner's area and the outcome-probability ranking's area, in the primary cells (95% interval, then estimate):

| Cell | X-learner minus outcome-probability ranking |
|---|---|
| Hillstrom men's, visit | -0.00340 to 0.00035; -0.00142 |
| Hillstrom women's, visit | -0.00003 to 0.00347; 0.00170 |
| Criteo, visit | 0.00003 to 0.00013; 0.00008 |

The same difference in the secondary conversion cells (reported, not judged):

| Cell | X-learner minus outcome-probability ranking |
|---|---|
| Hillstrom men's, conversion | -0.00079 to 0.00023; -0.00029 |
| Hillstrom women's, conversion | -0.00017 to 0.00072; 0.00028 |
| Criteo, conversion | -0.000032 to -0.000002; -0.000017 |

- Only in Criteo visit does the X-learner rank better than the outcome-probability ranking, and there the gain (0.00008) is small next to the area itself (0.00310).
- In the women's Hillstrom cell, which passes both marks, the X-learner beats random but not the outcome-probability ranking. The run logs this as "uplift modelling added nothing over targeting by risk on this dataset". The same note is logged for Criteo conversion, where the interval lies entirely below zero: there the X-learner ranks worse than the outcome-probability ranking.

### Sanity mark: naive difference and average predicted effect

Each learner's average predicted effect should fall inside the 95% interval of the naive treated-minus-control difference on the evaluation half.

| Cell | Naive difference, 95% interval | Estimate | T-learner average | X-learner average | Sanity mark |
|---|---|---|---|---|---|
| Hillstrom men's, visit | 0.06660 to 0.08590 | 0.07602 | 0.07660 | 0.07634 | passed |
| Hillstrom women's, visit | 0.03847 to 0.05588 | 0.04702 | 0.04369 | 0.04339 | passed |
| Criteo, visit | 0.00992 to 0.01077 | 0.01034 | 0.00733 | 0.00722 | **failed** |
| Hillstrom men's, conversion | 0.00420 to 0.00955 | 0.00695 | 0.00670 | 0.00672 | passed |
| Hillstrom women's, conversion | -0.00031 to 0.00395 | 0.00175 | 0.00447 | 0.00448 | **failed** |
| Criteo, conversion | 0.00106 to 0.00125 | 0.00116 | 0.00099 | 0.00096 | **failed** |

- **The sanity mark fails in the primary Criteo visit cell.** Both learners' average predicted effect lies below the naive difference's interval. This failure is not explained.
- It also fails in two secondary cells, Hillstrom women's conversion and Criteo conversion. These failures are not explained either.

## Run

- One run, 3,077.7 seconds, peak memory 3.90 GB, no swap used.
- Per-step timings (model selection, refits, bootstrap) were not logged. Only the time per bootstrap resample was: about 0.001 seconds for the Hillstrom cells and about 0.32 seconds for the Criteo cells.

## What this means

- The method beats random in two of the three primary cells, and the first Done-when criterion is not met: it needs all three, and the men's Hillstrom cell fails both marks.
- Where it beats random, it adds little or nothing over ranking by outcome probability.
- These results test the method on other companies' data. They say nothing about KKBox customers. No KKBox per-customer effect exists (see `docs/causal_assumptions.md`).

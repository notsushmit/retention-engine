# Causal assumptions for the KKBox uplift estimate

The KKBox result is, in practice, a comparison inside one month and one payment method. All 70 analysed discounted renewals are on payment method 23, and 68 of them are in September 2016. From January to November 2016, 1,578,643 of 1,582,100 renewals at full or discounted price (about 99.8%) were on payment methods with no discounted renewal at all that year. Those renewals came from 233,546 of 233,945 renewing customers. For them the data show no chance of a discount, so no effect can be estimated. The KKBox data therefore cannot tell us, in general, whether a discount keeps a customer. This is a finding about the data, not a failure of the method. The method is checked separately on two randomized datasets, Hillstrom and Criteo, where chance decided who received the action.

The design is registered in `configs/uplift.yaml` before any outcome was read. Every count on this page comes from the logged recount (`python -m retention.uplift recount`) on the 20% customer sample. The recount read transaction amounts, pre-snapshot label columns and pre-snapshot features, never an outcome, and it stops if any count differs from the registered value or from this page.

## What is estimated

- **Action (treatment):** a renewal paid below list price. A renewal counts as discounted when the list price is above zero, something was paid, and the amount paid is at least 5% below list. The comparison is a renewal at exactly the list price.
- **Outcome:** paid churn at the expiry that the renewal creates, meaning no paid renewal within 30 days of that expiry.
- **Group the estimate refers to:** payment-method-23 renewals of a 30-day plan in August and September 2016. That is 2 discounted against 308 full price in August, and 68 against 242 in September.
- **Quantity:** the average difference in churn between all of those renewals discounted and none of them discounted, estimated by augmented inverse-probability weighting (AIPW), which combines the outcome models with the propensity and stays correct if either one is right. The T-learner and X-learner averages are reported beside it. If trimming leaves about 25 full-price renewals, as expected, per-renewal estimates would be noise and are not reported as findings.

A discounted price appears only on a renewal, so every discounted case renewed at that event, and renewal there cannot be the outcome. The estimate says nothing about customers who did not renew at that event.

## How discounts were assigned: four hypotheses

Discounts were not randomly assigned, and the data do not record how they were assigned. Below are four hypotheses. The data cannot tell them apart: a renewal row records only a payment method number, a list price and the amount paid. A propensity model can show which features differ between discounted and full-price renewals, but that does not identify the mechanism.

1. **A campaign tied to one payment method and one month.** Every first renewal paid below list from January to November 2016 is on payment method 23, and in the analysed months they cluster in September. September 2016 is also the month of a known level shift, when new customers and free trials rose.
2. **Payment-method or partner pricing.** Method 23 may be a partner or channel with its own price list. The discount would then belong to the channel, not to an offer.
3. **Retention offers aimed at renewals that looked likely to lapse.** This would make discounts look harmful.
4. **Self-selection.** Customers who look for lower prices may differ in ways the data do not record.

What the counts show about assignment:

- Of the 68 discounted September renewals, 65 belong to customers who had a discounted method-23 transaction (list 149, paid 129) in the year before. That is true of 25 of the 242 full-price September renewals.
- The timing separates the two groups completely. For the discounted renewals, the earlier discount fell between November 2015 and August 2016. For the full-price ones, it fell in September or October 2015.
- This looks like a recurring billing pattern tied to earlier discounts more than a one-off offer, which fits hypotheses 1 and 2 better than 3 and 4. The data cannot confirm it.
- Because the recency of the last discount separates the groups completely, it is kept out of the propensity model; with it there would be no overlap at all. Its imbalance is reported as a diagnostic: it is evidence about how discounts were assigned, not a covariate.

## The assumptions

1. **No hidden confounding (exchangeability).**
   - Given the features known before the snapshot date and the snapshot month, whether a renewal was discounted is unrelated to the churn it would have had at either price.
   - This cannot be tested from the data. It fails if the assignment depends on something outside the features, such as an offer email, a partner contract or a support contact.
2. **Overlap (positivity).** Every renewal in the analysed group must have had a real chance of either price.
   - Renewals on other payment methods are outside the group.
   - Within method 23, July 2016 has no discounted renewal and October 2016 has no full-price one, so both months are dropped before any model.
   - The remaining overlap is checked with a propensity model (the estimated chance of a discount from the features), and extreme cases are trimmed by a rule registered in advance.
3. **No interference.** One customer's discount does not change another customer's churn. This is plausible for individual subscriptions. It would fail for shared or family accounts, which the data do not identify.
4. **Consistency (one well-defined treatment).** "Discounted" must mean the same thing throughout. In effect there is one discount level, 13.4% below list (69 of 70 analysed discounted renewals; the other is 33.6% below list). How the discount was delivered, and whether it lasted beyond one renewal, is not recorded.
5. **Pre-treatment covariates only.** The propensity model uses:
   - the frozen feature set fs1, each column built from rows dated before the snapshot date: listening in the 28 days before it, coverage and segment state, the last transaction, tenure and registration channel. Columns that are constant across the analysed renewals, and values held by fewer than 5 of them, are dropped;
   - the snapshot month (September or August);
   - whether the customer had any discounted renewal in the 365 days before the snapshot date. This was added after counting, before any outcome was read. The only discount column in fs1 (the discount on the last transaction) is zero for all 620 analysed renewals, while a past-year discount marks 65 of 68 discounted September renewals against 25 of 242 full-price ones.

   In addition:
   - Profile columns (age, gender, city and the blank-profile flag) are not used. The members table is a snapshot taken after the data period, and whether a customer has a row in it depends on later activity.
   - Nothing about the renewal after the snapshot date is a feature, apart from its own price, which defines the treatment.
   - Tenure and registration channel also come from the members table, but they are fixed at sign-up, so they are used, under the same masking as the churn model: null when the customer has no listening before the snapshot date, with a flag marking the null.
   - Registration channel may stand in for plan type or for how the customer pays, so part of what it adjusts for may be the assignment itself. A sensitivity run without tenure and registration channel is registered in advance.
6. **Outcome measured the same way in both groups.**
   - The outcome uses paid-based labels rebuilt from 2016 transactions only.
   - A same-day subscribe-and-cancel of the same plan does not undo the renewal; it only turns off auto-renew.

Age, gender and registration channel are never presented as something to act on.

## What the KKBox result can and cannot show

It can show, first of all:

- whether the data are large enough to give any estimate. At least 30 discounted and 30 full-price renewals must remain after trimming; otherwise the effect is reported as not estimable.

Registered expectation, written before any propensity model was fitted; it is not a result. With past discounts in the propensity model, the September full-price renewals without a past discount and nearly all August renewals are expected to fall outside the trimming band. That would leave about 65 discounted against about 25 full-price renewals, below the 30 full-price minimum, so "not estimable" is the expected answer. The result section reports what the propensity step actually found.

If that holds, it can also show:

- whether discounted and full-price method-23 renewals in August and September 2016 differ in churn after adjusting for the observed features;
- whether that difference holds up when the trimming rule changes;
- whether the placebo outcome shows a difference that should not be there.

It cannot show:

- that discounts change churn. That rests on assumption 1, which cannot be tested;
- anything about other payment methods, other months, other discount levels, or customers who did not renew;
- the effect of any other action, such as reminders, emails or calls. KKBox data contain no such actions. Hillstrom and Criteo come from other companies and other products, so they test the method, not KKBox customers. The budget step treats every action effect as a labelled assumption.

## Checks registered in advance

- **Placebo: low power.** With 70 and 550 renewals, the smallest difference the placebo detects with 80% probability at the 5% level is about 0.36 standard deviations of the placebo outcome (2.8 × √(1/70 + 1/550)). That is before trimming and weighting, which make it larger. If trimming leaves about 65 against 25, the figure becomes about 0.67 standard deviations (2.8 × √(1/65 + 1/25)). Either is a large difference, so a placebo near zero is more likely a lack of power than an absence of confounding. The placebo outcome's standard deviation is reported with the result.
- **Placebo: design.**
  - The same estimate is repeated with an outcome that ended before the snapshot date: days with listening in days 56 to 29 before it. This outcome is kept out of the features, so the real discount, dated after it, cannot have changed it.
  - This outcome was chosen because it lies just outside the features' 28-day listening window and is measured for every customer. Past listening, however, is only a weak stand-in for the unrecorded things that drive churn, such as an offer, a contract or a support contact.
  - A clearly nonzero placebo, with a 95% interval that excludes zero, is strong evidence of confounding: the two groups differ in a way the adjustment does not remove. The main estimate is then reported as failing its assumption check.
  - A placebo near zero is weak evidence. With 70 discounted renewals it may only mean the placebo was too imprecise to detect anything.
- **Trimming sensitivity.** The estimate is repeated with the primary trimming rule, with no trimming, and with a stricter rule. It is called sensitive if the sign changes, or if one rule's interval excludes zero while another's does not.
- **Discount margin.** Not used as a check. Every first renewal paid below list from January to November 2016 is either 149 to 129 (13.4% below list) or 149 to 99 (33.6%), so any margin from 1% to 13% gives the same discounted set and varying it changes nothing.
- **Implausible results.** If discounts appear to raise churn, with the whole interval on that side, this is reported as a sign that the assumptions fail, not as a finding.
- **Balance.** After weighting, every covariate must have a standardised mean difference of at most 0.10 in absolute value. The standardised mean difference is the gap between the two groups' averages, divided by the typical spread of that covariate. If any covariate exceeds the mark, the estimate carries the label "balance not met" and is not reported as a finding; the propensity model is not changed afterwards. With about 65 against 25 renewals, chance alone gives these differences a standard error near 0.24, so the mark may fail from noise.
- **Without tenure and registration channel.** The estimate is repeated without these two columns.
- **Too few cases.** If fewer than 30 discounted or fewer than 30 full-price renewals remain after trimming, the result is "not estimable". The outcome is then never read for KKBox: no effect estimate, no naive difference and no placebo is run, and no estimate is made on the untrimmed renewals, because some of them have no comparable renewal in the other group. In that case the robustness checks above exist in design, registered in advance, but were not run.

## Results

Pending. When this page was written, no propensity model had been fitted and no KKBox outcome had been read. Results are added here from logged runs only.

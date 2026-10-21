# Serving parity

Every check that the scoring service answers as batch scoring does, in the order the steps ran. Each figure comes from the run report named beside it (under the sample's serving folder) and the commit it ran at. Two registered marks are not met as written and are judged under amendments: A9 for the step 6 feature parity and A10 for the step 8 HTTP parity.

## Step 4: the scoring function against Phase 7 (step4_report.json, a3d0a07)

23 of 23 registered marks were met exactly on 2016-10-01 and 2016-11-01: the model reloads through the registry alias, the final-probability tie rule, the population counts, the unseen-level check, the reasons, the paid days, the action blocks and the no-price rule.

## Step 6: the replayed store against batch

| Check | Registered | Result | Report, commit |
| --- | --- | --- | --- |
| October 2016 events | 3,727,472 | recount 3,727,472; produced 3,727,472; consumed 3,727,472: met | recount_2016-10.json, 3526627; replay_full.json, 7a71e08 |
| Feature parity, as written | every column of the feature table (68) | not met as written (mis-specified before any run) | amendment A9, written before the result |
| Feature parity under A9 | 61 columns, 230,286 rows | 230,286 rows compared, no column differs: met | stream_parity_2016-11-01.json, 7a71e08 |
| Score parity | 169,673 rows, final probabilities within a relative 1e-12 | 169,673 rows, largest gap 0.0: met | stream_parity_2016-11-01.json, 7a71e08 |
| Determinism | full speed and throttled give the same table hash | both 1a561a3f...04bf: met | replay_throttled.json, 153c803 |

A9 was written while planning the move, before any move code or run, and registered before the parity run produced its result; unlike A10, it was not written after a result was seen.

The full-speed replay took 550.6 s and peaked at 1.671 GB. The throttled replay took 816.8 s and peaked at 2.145 GB; it started 6 minutes after a restart of the Mac, so these figures are not comparable, and determinism compares the table hash only.

## Step 8: the stack over HTTP

| Check | Registered | Result | Report, commit |
| --- | --- | --- | --- |
| Reload of the store | 7 counts and 230,294 keys, exact | scored 169,673; trial only 8,890; promo only 31; beyond 30 days 51,692; with coverage 230,286; not active 195,234; known 425,520; keys 230,294: met | load_snapshot_2016-11-01.json, 224f672 |
| HTTP parity, as written | 1,000 of 1,000 answers equal to the host's as exact strings | 996 of 1,000: not met as written | http_parity_2016-11-01_224f672.json, 224f672 |
| HTTP parity under A10 | 1,000 of 1,000 with every non-float field exact and every float within a relative 1e-12 | 1,000 of 1,000: met | http_parity_2016-11-01.json, 85ce0f1 |
| Latency (p95 on this Mac, one run of 1,000 calls) | p95 at most 100 ms (my unvalidated choice) | p95 69.429 ms: met | latency_2016-11-01.json, 85ce0f1 |
| Concurrency | 8 threads, every answer equal to its sequential answer, no 5xx | 1,000 /score and 50 /at-risk calls, 0 differing, 0 server errors: met | http_checks_2016-11-01.json, 85ce0f1 |
| December 2016 over HTTP | 409, no reasons, a clean scan | 409, no reasons, 0 scan hits: met | http_checks_2016-11-01.json, 85ce0f1 |
| Served config | equal to the committed file | equal: met | http_checks_2016-11-01.json, 85ce0f1 |
| Registry after startup | no file written or migrated | 26 files, the same tree hash and the same database hash before and after: met | http_checks_2016-11-01.json, 85ce0f1 |

The registered HTTP mark compared each answer as an exact string, and it was not met: 996 of 1,000 matched at 224f672, and that result stays on record. A diagnosis that was not a mark (http_parity_diagnosis_2016-11-01.json, b37be77) asked the container and the host twice each: each side gave the same answers both times, and the 4 differing customers differed only in float values, 2 to 4 ulps apart (the smallest steps a float can take), with a largest relative gap of 6.65e-16; no text, action name or flag differed. The likely source, untested, is the container's Linux libraries against the Mac's. Amendment A10 was written after that result was seen. It judges floats within a relative 1e-12 and every other field exactly; under it all 1,000 pass. A last-bit gap could in principle flip a threshold for customers outside the sample; inside the sample such a flip would fail A10, because discrete fields stay exact.

The latency run also gave p50 59.242 ms, p99 97.809 ms, a mean of 61.065 ms and a maximum of 108.142 ms, over loopback with one client and one connection. It ran before the API container had its 1.5 GiB memory limit. The HTTP parity compares the container and the host running the same scoring code: it shows that the HTTP path and the container add no difference beyond the float gaps above, not that the scoring is right, which steps 4 and 6 cover.

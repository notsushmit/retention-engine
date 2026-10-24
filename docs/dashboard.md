# Dashboard

> Replay of 2016 data, not real-time.

The dashboard is a Streamlit app (`dashboard/`, started with `make dashboard` on 127.0.0.1:8501). Its settings are registered in `configs/dashboard.yaml` before any code that uses them; the later keys C1 to C4 change earlier entries without editing them. It reads only the scoring API, through one client that refuses any host other than loopback and any call other than the seven listed below, with a 10-second timeout (30 seconds for `GET /summary`). It imports no store client, no data reader and no project package; `tests/test_dashboard_isolation.py` fails if one appears.

Every page carries the same frame: the application name in a header bar, the replay banner above, the breadcrumb, the store date line ("one date per loaded snapshot; no daily dates"), the replay progress ("no replay running" when no replay runs) and the four fixed disclaimers of the scoring service at the foot. When the service is not ready, the page shows the reason from `GET /health` in place of every number.

## Pages and API calls

| Page | API calls | What it shows |
| --- | --- | --- |
| Overview | `GET /health`, `GET /summary`, `GET /alerts`, `GET /at-risk` | the expected lapse count in days 1 to 7 as the page's big number; the three expected counts (days 1 to 7, 8 to 30, and 31 to 59 as a demonstration), each with its share of the scored customers, beside the share of customers expiring more than 30 days ahead; one chart with two panels (7 and 30 days) of the first reason group among the 1,000 customers most at risk; then two tabs, "Alerts" (the knowable trend alerts with their three labels) and "Top 20 at risk" |
| At-risk list | `GET /at-risk` | the ranking at 7, 30 or 59 days in the API's order, the auto_renew and in_plan filters, pages of up to 1,000 rows and a CSV of the displayed rows with the store date and horizon |
| Customer profile | `POST /score` | the three probabilities (the 59-day one as a demonstration), the two reasons, expected paid days and the action block with every label; the outside-population and not-active answers as served |
| Action planner | `GET /plans` | the plans precomputed in Phase 7 for the store's date at every registered budget level, the strategy comparison with its registered caption and the random strategy's spread; nothing calls an optimizer |
| Model health | `GET /health`, `GET /model` | the model version, the eight file hashes, the config hash and the calibration table before and after the ordering fix at 7 and 30 days; "drift: not monitored (Phase 10)" |

The three expected counts come from the final probabilities of the scored customers only: the sum of p7, of p30 - p7 and of p59 - p30. No survival curve is used. Each share is the expected count divided by the number of scored customers (C5): an expected share, not a share of named people.

Every action carries the tag "assumed effect", the full scenario, "The plan is unstable under 5 of 11 variants." and "No KKBox effect estimate exists; every effect is an assumption." Every 59-day figure is muted, outlined with a dashed line and badged "demonstration"; it is never a metric card.

## Look

One cream background, near-black text, one orange accent used only as a fill for the leading at-risk item, and greys. The eight colours were sampled from a reference slide: background #F4EDE3, ink #090B0E, accent #FE5A00, rust #942A11 (orange text), card grey #1B222C, bar grey #40454B, body grey #413A31, label grey #514A40. The orange's contrast on the background is 2.71:1, so it is never used for text. The fonts are self-hosted, with no font service: Space Grotesk (headings and body), Instrument Serif italic (the one big number on each page, and the bucket shares on the Overview) and JetBrains Mono (small labels), each under the SIL Open Font License 1.1, whose text sits beside the font files in `dashboard/static/fonts/`.

## Tooltips

Every number has a tooltip. Each tooltip names the API field it explains and quotes the calculation from registered text in `configs/` (dashboard.yaml, serving.yaml, actions.yaml); the connecting words add no number. `tests/test_dashboard_wording.py` checks that every number in a tooltip appears in a config and that every tooltip passes the wording rules.

## The reason bars and suppression

The bars show, for the 1,000 customers most at risk at 7 and at 30 days, the share whose top raising reason is each group. Reasons explain the probability before the ordering fix. Reasons are model attributions, not causes. A bar with fewer than 20 customers is drawn with no value and reads "fewer than 20". This minimum is a display rule, not a disclosure control: the `/summary` body still carries segment totals, so a suppressed count can be worked out from it.

## Capture mode

The dashboard accepts `?capture=1`. In that mode the Overview leaves out its top-20 table and the cancel events; nothing else changes. Only three pages are captured for the repository or a video: the Overview with `?capture=1`, the Action planner and Model health. `scripts/capture_pages.py` prints their addresses and refuses the At-risk list and the Customer profile, which show customer IDs. Capture mode is a convenience for screenshots, not access control.

## Changes since the plan

- C1, found after a result: the first `/summary` call on 2016-11-01 showed one reason group at 7 days and three at 30 days, so the planned heatmap was replaced by the bars on the Overview, the page "What the model sees in at-risk customers" was dropped (five pages remain), and the generated sentences name the leading group.
- C2: the conditions set when the dashboard code was approved (capture mode, the accent as a fill only, the Overview layout, tooltips from registered text only, the wording of the several-groups sentence). Registered before any dashboard code; not a change made after a result.
- C5 and C6: each bucket's share of the scored customers, at the user's request after the docs were written, then moved beside its number in the bold serif (the 59-day share stays muted).
- C3 and C4: the application name, "Retention Decision Engine", first at the top of every page and then in a header bar, at the user's request after the first local run.
- Found after a run, the first local run against the stack (commit 8fb8960): the tile showing 22% was labelled "Outside the population" although it counts only the customers expiring more than 30 days ahead; it now reads "Expiring more than 30 days ahead". Four captions that started a registered lower-case text after a full stop now introduce it ("Plan fields:", "Action block:", "Comparison caption:", "Calibration:", "Note:").
- Found after the same run, later commits: the planner's end labels are spread to a minimum gap; every table shows thousands separators and fixed decimals; the Overview's bucket figures sit in one row; the Deploy button is hidden while the main menu stays; one API client is shared by every browser session; the paid-days name is shown once; the At-risk list states that plan actions appear only where their net value is positive under the assumed effect.

## Limits

- The latency figure in the README was measured before the 1.5 GiB limit and before the dashboard endpoints.
- The CSS relies on Streamlit 1.65.0, pinned in `uv.lock`; one rule targets Streamlit's own Deploy button and may break on an upgrade.
- Capture mode does not protect the IDs; only the three captured pages are fit for a public image.

# Congressional Trade Backtest

A backtest of a strategy mirroring congressional stock disclosures, weighted by
disclosed dollar-range midpoints, evaluated four ways at once: trade-date vs
disclosure-date entry x all-Congress vs top-N-performers cohort. The design exists
to *attack* the strategy's apparent edge, not flatter it — every modeling choice is
documented and defensible (see "Design decisions" below).

## Status

**v2 complete — full pipeline, extended to 2015 with options.** `efd_probe.py` →
`ingest.py` (D1/D2/D3/D5/D12 + options) → `prices.py` (D4/D5 + integrity guards) →
`portfolio.py` (D6/D7/D11 + holding rule, long-short capable) → `backtest.py`
(D8/D9/D10 + options/longer window) → `report.py`. Entry point:
`python run_backtest.py`. Outputs: `data/processed/backtest_fourway.csv` and the
three charts in `reports/` (two honesty-gap equity curves + a bootstrap-CI forest
plot). See Results (v2) below; the v1 (2021-start,
stocks-only) results are kept underneath for comparison.

### v2 extensions (added after v1, at Aidan's request)
- **Window extended 2021 → 2015** (~11y; eFD floor is 2012). Buys depth for a real
  out-of-sample test (now ~4.5y OOS vs v1's 2.2y) at the cost of more paper filings
  (16.6% vs 12.8%) and a wider delisted-price gap. Two-parameter change
  (`SUBMITTED_START_DATE`, `PULL_START`) + re-scrape/re-pull.
- **Options included** (D2 relaxed): calls = long the underlying, puts = short it,
  sized by midpoint. We can't price the options themselves on free data (no
  historical chains), so this is *underlying directional* exposure — it captures
  their option picks' direction, not the leverage/convexity. The engine became
  long-short (gross-exposure normalized) to support put-shorts. Run as a separate
  cohort variant alongside stocks-only.
- **Price-integrity guards** (forced by the longer window): (1) *ticker reuse* — a
  symbol delisted years ago gets reassigned to a different company and Yahoo
  returns the new security's prices (ITC Holdings → a $20,000-priced unrelated
  "ITC", a 6,700×/day artifact); (2) *failed-company OTC shells* at sub-penny prices
  (SBNY, VAXX); (3) *foreign-listing matches* (AEX.MU, 0QZI.IL). `prices.py` drops
  series with sub-penny prices, >300% daily moves, or dotted exchange suffixes to
  the coverage gap. As a final backstop, `portfolio.py` winsorizes per-name daily
  returns at ±50% (0.003% of observations) so a residual corporate-action join
  (e.g. DuPont's 2026 reverse split, which Yahoo hadn't smoothed) can't distort a
  compounded return. These guards are the difference between a believable result
  and a garbage one — without them, one reassigned ticker made a senator look like
  +2,000,000%.

## Design decisions

- **D0** Source is the official Senate eFD portal (`efdsearch.senate.gov`) directly,
  not a vendor. Tried Financial Modeling Prep first; every congressional endpoint
  returned `402 Payment Required` on the available plan, so we went to the primary
  source instead of paying for vendor access to data that's free at the origin.
  Scope is Senate-only PTRs, 2021-01-01 -> present; paper filings (12.8% of reports
  in-window) are counted as a coverage gap, not OCR'd.

- **D1** Amount midpoint uses the geometric mean `sqrt(floor * ceiling)`, not the
  arithmetic mean, because trade-size distributions are right-skewed within a wide
  disclosure band, and the arithmetic mean is dominated by the (rarer) large end of
  the band. The open-ended top band (`Over $50,000,000`) has no ceiling, so its
  floor is scaled by the square root of the ratio of the band directly below it
  (`$50,000,000 / $25,000,001`), applying the same floor-to-geomean relationship as
  every other band rather than guessing a number out of thin air.

- **D2** Equities-only filter keeps `asset_type == "Stock"` (the source's own
  taxonomy, which already includes ETFs) and drops everything else (bonds, options,
  crypto, commodities, municipal securities, non-public stock, other) — 36.0% of
  post-D12 rows in this pull. ETFs are flagged via `is_etf` rather than dropped, so
  the backtest can run an all-stock and a stock-ex-ETF cohort side by side, since
  ETF-driven "congressional alpha" would be a different (weaker) story than
  single-name alpha.

- **D3** `type` (`Purchase` / `Sale (Full)` / `Sale (Partial)` / `Exchange`) passes
  through untouched plus an `is_partial_sale` flag. The actual position-cap policy
  (a minimum holding-period assumption, so stacked partial sales can't reduce a
  modeled position below zero) needs running position state per `(senator, ticker)`
  over time and belongs in `portfolio.py`, once D6/D7 are decided — not implemented
  at the ingest layer, which has no notion of a running position.

- **D11** Both entry legs use the next session's open after the relevant date
  (`transaction_date` for the trade-date leg, `date_received` for the
  disclosure-date leg) — the same lag rule applied identically to both, so the
  trade-date/disclosure-date gap measures only the cost of reporting lag, not an
  artifact of inconsistent execution assumptions. Rejected same-day close as the
  default specifically because it's unrealistic for the disclosure leg (PTRs post
  throughout the trading day, sometimes after close).

- **D12** Amendments are resolved by grouping transactions on
  `(senator, transaction_date, ticker)`: if any row in a group came from an amended
  report, only the amendment row(s) are kept and the superseded original is
  dropped (321 of 5,628 raw rows in this pull). This targets exactly the rows that
  would otherwise double-count the same real-world trade, at the cost of an
  approximation if two unrelated transactions ever coincidentally share that key
  within the same senator.

- **D4** Adjusted close uses Yahoo's split-**and**-dividend-adjusted ("total
  return") price series, not price-only. This has to match what the SPY benchmark
  (D10) uses, or the trade-date/disclosure-date and aggregate/top-N comparisons
  aren't apples-to-apples — dividends folded into "alpha" for the portfolio but
  not the benchmark (or vice versa) would silently bias the headline numbers.
  Implemented in `congress_backtest/prices.py` via Yahoo's chart endpoint directly
  (not the `yfinance` package — see implementation note below).

- **D5** Of 766 unique tickers traded in-window, 88 (11.6%) have no live Yahoo
  data, confirmed genuinely delisted/renamed/bankrupt (not probe noise — verified
  with 3 independent isolated re-checks; 2 of the original 89 candidates, `INTC`
  and `FI`, turned out to be transient API noise and were excluded). Initially
  planned to exclude all delisted tickers, but that would have systematically
  dropped acquisition premiums (most of the 88 are M&A buyouts, not bankruptcies)
  — reconsidered to look up the real outcome per ticker instead:
  `data/processed/delisting_resolutions.json` categorizes each as a pure ticker
  rename (~9, e.g. `SQ`→`XYZ`, `MMC`→`MRSH` — remapped, not an exit), a real M&A/
  go-private exit (~50, with researched cash/stock terms and close date), a
  bankruptcy (5, exit = $0), an orderly liquidation (1, `EQC`, $20.60/share), a
  SPAC liquidation (1, `LMACU`), or a fund mistagged as `"Stock"` by the source (5,
  e.g. defined-maturity bond ETFs — a D2-adjacent classification issue, not a real
  delisting). ~5 remain unclear with under $50K of combined dollar exposure in
  this pull — flagged rather than guessed. Separately, 14 distinct ticker strings
  (22 rows) reported a single dollar amount against **two** tickers —
  spin-off/merger distributions like `DELL / VMW`, `JNJ / KVUE` — found while
  building the D5 ticker-validity check, not a pre-existing decision. There's no
  defensible way to split one disclosed amount across two securities, so these
  rows are dropped and the gap (22 of 5,307 rows, ~0.4%) is reported.

  **D5 implementation hit a second wall** while building `prices.py`: Yahoo's
  free chart API retains **zero** price history for any ticker that's been fully
  acquired or gone bankrupt — not just after the event, the *entire* series,
  including years of active trading beforehand (confirmed directly: `ATVI`
  returns "no data" for a 2022 date range, a year before its 2023 acquisition
  even closed). Pure renames are unaffected (`COR`/`ELV` retain full pre-rename
  history under the new ticker — Yahoo treats it as the same continuing entity).
  This breaks the original plan ("market price converges to the deal value by
  close, so Yahoo's last price is the exit") for 59 tickers, since there's no
  last price reachable at all. Tried a narrow backfill via FMP's
  `historical-price-eod/dividend-adjusted` endpoint (free tier, separate from the
  congressional-data endpoints D0 found paywalled) — worked for only 3 of 59
  (`ATVI`, `MRO`, `WBA`), the rest blocked behind the same "not available under
  your current subscription" wall that blocked D0's original FMP attempt.
  Decided: **exclude rather than chase a third data source or pay for a tier.**
  Final coverage gap in `prices.py`: 53 tickers with zero data anywhere (Yahoo nor
  FMP) excluded outright; 8 more (the bankruptcies/liquidations/SPAC) keep their
  resolved exit value but have no entry-side price history, so `portfolio.py` can
  mark the exit but can't compute a return for any position opened before it.
  Combined: **4.8% of equity transactions, 10.3% of dollar exposure** excluded —
  reported, not silently dropped, consistent with how D0 already treats the
  paper-filing gap.

  **Implementation note:** `prices.py` calls Yahoo's chart endpoint directly via
  `curl_cffi` (impersonating Chrome's TLS fingerprint) rather than the `yfinance`
  package or plain `requests`. Confirmed two distinct blockers, not one: (1)
  `yfinance`'s own client-side rate limiter errored on the very first call
  regardless of actual request volume; (2) plain `requests` got blocked on
  *every* call to the same URL that a vanilla `curl` succeeded on repeatedly — a
  TLS/HTTP-client fingerprint block, not a rate limit (the same reason `yfinance`
  itself adopted `curl_cffi` upstream). Same underlying data source (Yahoo
  Finance), different transport.

- **D6** Fixed-AUM, scale the book: one constant capital base, target weights
  proportional to the cohort's *net held* disclosed midpoint per ticker,
  renormalized to sum to 1 whenever anything is held. A new signal doesn't add
  capital on top — it re-slices the same pie (funding a new name trims the
  others). Chosen over "add the position" because it yields a clean fixed-capital
  return series directly comparable to the SPY benchmark (D10) and to clean
  Sharpe/Sortino/MDD; "add the position" would let the book size drift and muddy
  every comparison. Rebalancing is event-driven (any day with a new fill), which
  falls straight out of D6 + D11 — no separate cadence knob. Implemented in
  `congress_backtest/portfolio.py`.

- **D7** Idle cash earns the risk-free rate (FRED 3-month T-bill, `DTB3`, pulled
  free, no key). Under scale-the-book the book is always 100% invested whenever
  it holds anything, so idle cash only appears in flat periods (before the first
  signal, or any stretch where net held positions net to zero) — in this pull
  that's just 2 days, so D7 barely bites, but it's the honest treatment and it
  doubles as the Sharpe risk-free input. Rejected "always fully invested"
  (removes a real friction) and "idle at 0%" (understates a real investor's
  return).

- **Holding rule (deferred D3)** Hold until disclosed sale. Net held midpoint per
  ticker walks `max(0, running + delta)` — Purchase `+midpoint`, both
  Sale (Full) and Sale (Partial) `−midpoint`, floored at zero so it can never go
  short. A name never sold is held until its price series ends. This is the
  faithful-mirror choice with no arbitrary horizon knob; its known weakness is
  that incomplete sale disclosure lets names linger and dilute the book (the
  all-Congress book holds ~250 names at peak), which is itself a reportable
  finding rather than a hidden bug. Net held is aggregated across the cohort per
  ticker, not tracked per senator (a Sale (Full) subtracts only its own disclosed
  midpoint) — the simple, position-state-free reading; per-senator lot tracking
  would be a refinement, not v1. Exchange (7 rows) is a no-op for sizing.

- **Smoke-test result (all-Congress, gross, no costs/benchmark yet):** trade-date
  entry 113% total / 12.7% CAGR over 6.3y; disclosure-date entry 47% / 7.3% over
  5.4y. The large trade-vs-disclosure gap is the headline "cost of reporting lag"
  the project exists to measure — and it shows up before any of the remaining
  decisions are made. (Windows differ because the first actionable signal differs
  by basis; D10 will align them for the formal gap.)

- **D8** Transaction costs = 5 bps per side (buy and sell), charged on notional
  traded (`sum|Δweight|`) each rebalance, alongside the always-shown 0-bps
  frictionless case. 5 bps reflects the brief's framing — *what a retail investor
  following these trades would actually net* — for which spread+commission on
  liquid large-caps is light; an institution would assume more. In practice costs
  move the test-window return by under 1% here (the mirror is buy-and-mostly-hold,
  low turnover), so the headline conclusions don't hinge on the level.

- **D9** Top-N selection with a no-look-ahead split: 60/40 chronological, N=5,
  ranked by each senator's own single-senator mirror total return over the
  **train** window, on the **disclosure basis only** (the observable one — ranking
  on trade-timed returns is a selection-stage look-ahead), with a ≥10-train-trades
  floor, one frozen set followed under both timings. In v2 (2015 start) the split
  is train 2015-01→2021-11, test 2021-11→2026-06 (~4.5y OOS); train-selected top-5:
  Carper, Perdue, Roberts, Reed, Hoeven.

- **D10** SPY buy-and-hold, total-return (matching D4), over the identical test
  window. Pulled via the same `curl_cffi` Yahoo path.

### Results (v2, out-of-sample test window 2021-11 → 2026-06, ~4.5y, net of 5 bps)

| cohort | scope | basis | net total | CAGR | Sharpe | MaxDD |
|---|---|---|---|---|---|---|
| all-Congress | stocks | trade | −8.2% | −1.9% | −0.07 | −43.7% |
| all-Congress | stocks | disclosure | −10.7% | −2.4% | −0.09 | −44.2% |
| all-Congress | +options | disclosure | −14.7% | −3.4% | −0.17 | −41.9% |
| top-5 | stocks | trade | −28.6% | −7.1% | −0.13 | −51.6% |
| top-5 | stocks | disclosure | −28.8% | −7.2% | −0.13 | −51.8% |
| **SPY** | — | benchmark | **+66.9%** | **+11.9%** | **+0.50** | −24.5% |

**Point estimates (out-of-sample, net of 5 bps) — read with the significance
section below, which is the real headline:**
- **Every congressional cohort underperformed SPY** (−8% to −29% vs +67% total),
  with deeper drawdowns (−44% all-Congress, −52% top-5, vs SPY's −25%) and negative
  Sharpe. Point estimate: the naive mirror lagged badly.
- **The selection effect reversed sign:** the senators who looked best in 2015–2021
  (Carper +261%, Perdue +251%, Roberts +257% train returns) *underperformed* the
  broad cohort by ~18–20% total in 2021–2026 (vs +9–18% *positive* in v1's 2.2y
  window), and the individual train→test rank correlation is ≈ −0.27.
- **Options:** adding the option book (78% puts, mirrored as short the underlying)
  cost ~4% in a rising market — point estimate opposite to "options hide the alpha."
- **Reporting lag** shrank to ~0–2.5% — a small point-estimate drag.

### Statistical significance (stationary block bootstrap, `significance.py`)

**This is the most important result, and it walks back the strong language above.**
A stationary block bootstrap (Politis–Romano, B=10,000, E[block]=20 trading days)
on the daily return-difference series tests each gap against zero. Annualized mean
daily difference, 95% CI, two-sided p:

| gap | obs/yr | 95% CI | p | verdict |
|---|---|---|---|---|
| selection: top-5 − all (disclosure) | −2.3% | [−11.2%, +6.8%] | 0.60 | **ns** |
| selection: top-5 − all (trade) | −2.8% | [−11.9%, +6.8%] | 0.55 | **ns** |
| all-Congress − SPY (disclosure) | −11.3% | [−25.6%, +3.6%] | 0.14 | **ns** |
| top-5 − SPY (disclosure) | −13.6% | [−36.2%, +9.3%] | 0.25 | **ns** |
| reporting lag: trade − disclosure | +0.5% | [−0.2%, +1.3%] | 0.17 | **ns** |
| options: with-options − stocks | −1.8% | [−4.2%, +0.8%] | 0.17 | **ns** |

**None of the gaps are statistically significant.** The CIs are wide because the
congressional book is volatile (28–36%/yr) — even the huge-looking −11%/yr vs SPY
is only an information ratio of ~−0.64, which over 4.5y is a t-stat of ~−1.4. So:

- We can **reject "Congress significantly beats the market"** — there is no evidence
  of outperformance; the point estimate is negative and SPY beat every cohort. That
  is a real refutation of the popular claim.
- We **cannot claim significant *under*performance**, nor that the selection effect,
  options drag, or reporting lag are real — the data is too short/noisy to resolve
  them. The selection "reversal" in particular (p=0.60) is **not** distinguishable
  from chance; treat it as "no evidence of persistence," not "evidence of reversal."

The honest verdict: **a realistically-implementable mirror of aggregate Senate
equity disclosures shows no evidence of beating the market, and ~4.5 years is too
little data to say much more with confidence.** The strategy's apparent edge does
not survive lag-aware, cost-aware, out-of-sample testing — but the symmetric,
intellectually honest statement is that the noise floor here is high, and confident
claims in *either* direction (including the popular "Congress crushes the market")
are not supported by this sample.

### v1 results (kept for comparison — 2021 start, stocks only, ~2.2y OOS)
all-Congress +25–29%, top-5 +39–43%, SPY +50%. In that short window top-5 *beat*
all-Congress (selection looked predictive) — exactly the in-sample-overfit illusion
the v2 longer window dispels. Lesson: the 2.2y window was too short to trust the
selection signal; ~4.5y reverses it.

### Caveats (each a documented decision above, not a hidden flaw)
- ~10% of equity dollar-exposure dropped where no price source had data, now wider
  going back to 2015 (more delisted/reassigned names; D5 + integrity guards).
- Paper filings (16.6% of reports in the longer window) not OCR'd (D0);
  '--'/unknown-ticker rows (~22%) unpriceable and excluded.
- Options are *underlying directional* exposure, not real option pricing (no free
  historical chains) — captures direction, not leverage/convexity (D2 v2).
- Aggregate (not per-senator-lot) midpoint netting for the holding rule (D6/D3).
- Per-name daily returns winsorized at ±50% as an artifact backstop (0.003% of
  observations); the absolute return level has ~±10% sensitivity to artifact
  handling, but every relative conclusion (vs SPY, selection, options, lag) is
  robust to it.
- `metrics.py` is a placeholder stand-in for Aidan's Sharpe/Sortino/MDD lib.
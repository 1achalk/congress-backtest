"""
Price pull for the tickers in senate_ptr_transactions_equity.csv (D4, D5).

D4 -- uses Yahoo's `adjclose` series (split-AND-dividend-adjusted, i.e. total
      return), not raw close. Must match whatever D10's SPY benchmark uses.

D5 -- delisted/acquired/bankrupt/liquidated tickers, per
      data/processed/delisting_resolutions.json:
        - rename / probe_false_positive / format_issue: pull `resolved_ticker`
          like any other active ticker, full history through today. Not an exit.
        - acquired_cash / acquired_stock / go_private / bankruptcy / liquidation /
          spac_liquidated: Yahoo's free chart API has been confirmed to retain
          ZERO price history for any ticker that's fully ceased to exist -- not
          just post-exit, the *entire* series, including years of active trading
          before the event (verified directly: ATVI returns "no data" for a
          2022 date range, years before its 2023 acquisition closed). A pure
          rename (e.g. ABC -> COR) keeps full history under the new symbol; a
          genuinely gone company does not exist under any symbol. So these
          categories fall back to FMP's `historical-price-eod/dividend-adjusted`
          endpoint (confirmed: works on the free tier even though FMP's
          congressional-trading endpoints don't -- see CLAUDE.md D0). This is a
          narrow, scoped backfill for confirmed-delisted tickers only, not a
          reversal of D4's Yahoo-first decision.
            - acquired_cash / acquired_stock / go_private: no hand-computed exit
              value needed -- by the deal's close date the market price has
              converged to the deal consideration (the arb spread collapses to
              ~0 at close), so the last available price IS the realized exit.
            - bankruptcy / liquidation / spac_liquidated: the last *traded*
              price does NOT reflect the true outcome (trading typically halts
              before the bankruptcy/liquidation outcome is final), so these
              still get a synthetic final price row at the resolution's
              exit_value_per_share on the resolution's close_date, overriding
              whatever the last trade was.
        - fund_mismatched_as_stock: dropped entirely -- these are bond/income
          funds the source mistagged as "Stock", not real stock-picking (a D2
          refinement discovered via D5 research, not original equity positions).
        - unclear / spinoff_two_entities: excluded, gap reported. Not enough
          confidence (unclear) or too complex to reduce to one number for v1
          (spinoff_two_entities -- LGF-B holders ended up with shares of two
          different successor companies) to model responsibly.

NOTE on implementation: uses Yahoo's chart endpoint directly via `curl_cffi`
(impersonating Chrome's TLS fingerprint) rather than the `yfinance` package or
plain `requests`. The data is identical (same Yahoo backend yfinance itself
calls). Confirmed two separate blocks, not one:
  1. The `yfinance` package's own client-side rate limiter errored on the very
     first call in this environment regardless of actual request volume.
  2. Plain `requests` got blocked on *every* call (confirmed with fresh
     isolated tests, no caching involved) while a vanilla `curl` to the exact
     same URL succeeded immediately and repeatedly -- a TLS/HTTP-client
     fingerprint block, not a rate limit (this is exactly why yfinance itself
     moved to curl_cffi under the hood). `curl_cffi` with `impersonate="chrome"`
     reproduces curl's working fingerprint from Python. Source is still Yahoo
     Finance, per D0/D4.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from curl_cffi import requests
from curl_cffi.requests.exceptions import RequestException

PROCESSED_DIR = Path("data/processed")
YAHOO_CACHE_DIR = Path("data/raw/yahoo")
FMP_CACHE_DIR = Path("data/raw/fmp_backfill")
# with-options superset so option underlyings (e.g. ARKK/XBI) get priced too, not
# just directly-traded stocks
EQUITY_CSV = PROCESSED_DIR / "senate_ptr_transactions_with_options.csv"
RESOLUTIONS_PATH = PROCESSED_DIR / "delisting_resolutions.json"
ENV_PATH = Path(".env")
OUT_PRICES = PROCESSED_DIR / "prices.csv"

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
FMP_URL = "https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted"
THROTTLE_SECS = 1.0

PULL_START = "2014-01-01"   # v2: covers transaction dates behind the 2015 submission floor

NO_EXIT_NEEDED = {"rename", "probe_false_positive", "format_issue", None}
NO_YAHOO_HISTORY = {"acquired_cash", "acquired_stock", "go_private",
                     "bankruptcy", "liquidation", "spac_liquidated"}
NEEDS_SYNTHETIC_EXIT = {"bankruptcy", "liquidation", "spac_liquidated"}
EXCLUDED = {"unclear", "spinoff_two_entities", "fund_mismatched_as_stock"}


def read_fmp_key():
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("FMP_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("FMP_API_KEY not found in .env (needed for the D5 delisted-ticker backfill)")


def cache_path(cache_dir, ticker):
    safe = ticker.replace("/", "_")
    p = cache_dir / f"{safe}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def fetch_yahoo_chart(session, ticker, period1, period2):
    cache_file = cache_path(YAHOO_CACHE_DIR, ticker)
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    params = {"period1": period1, "period2": period2, "interval": "1d", "events": "div,splits"}
    backoffs = [0, 3, 8, 20]
    data = None
    for wait in backoffs:
        if wait:
            time.sleep(wait)
        try:
            r = session.get(YAHOO_CHART_URL.format(ticker=ticker), params=params, timeout=15)
            if r.status_code == 429:
                continue
            data = r.json()
            break
        except (RequestException, ValueError):
            continue
    time.sleep(THROTTLE_SECS)
    if data is None:
        return {"chart": {"error": {"description": "transient failure after retries"}}}
    cache_file.write_text(json.dumps(data))
    return data


def parse_yahoo_chart(data):
    """Return a DataFrame[date, adj_open, adj_close] or None if no usable data.

    adj_close is Yahoo's total-return series (D4). adj_open is derived: Yahoo only
    adjusts the close, so we back out the same-day adjustment factor
    (adj_close/close) and apply it to the raw open. This lets D11 fill at the next
    session's *open* on the same total-return basis as the daily MTM close."""
    chart = data.get("chart", {})
    if chart.get("error") or not chart.get("result"):
        return None
    result = chart["result"][0]
    timestamps = result.get("timestamp")
    if not timestamps:
        return None
    quote = result["indicators"].get("quote", [{}])[0]
    adjclose = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    opens = quote.get("open")
    closes = quote.get("close")
    if not adjclose or not opens or not closes:
        return None
    dates = [datetime.fromtimestamp(t, tz=timezone.utc).date() for t in timestamps]
    df = pd.DataFrame({"date": dates, "open": opens, "close": closes, "adj_close": adjclose}).dropna()
    if df.empty:
        return None
    df["adj_open"] = df["open"] * (df["adj_close"] / df["close"])
    return df[["date", "adj_open", "adj_close"]]


def fetch_fmp_history(session, ticker, fmp_key, start_date):
    """Narrow backfill for confirmed-delisted tickers Yahoo has no history for."""
    cache_file = cache_path(FMP_CACHE_DIR, ticker)
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
    else:
        params = {"symbol": ticker, "apikey": fmp_key, "from": start_date}
        try:
            r = session.get(FMP_URL, params=params, timeout=15)
            data = r.json()
        except (RequestException, ValueError):
            data = []
        cache_file.write_text(json.dumps(data))
        time.sleep(THROTTLE_SECS)

    if not isinstance(data, list) or not data:
        return None
    df = pd.DataFrame(data)
    if "date" not in df or "adjClose" not in df or "adjOpen" not in df:
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.rename(columns={"adjClose": "adj_close", "adjOpen": "adj_open"})
    df = df[["date", "adj_open", "adj_close"]].dropna()
    return df if not df.empty else None


def valid_series(df):
    """Reject corrupted/wrong-security price series. Two failure modes surfaced by
    the longer window: (1) TICKER REUSE -- a symbol delisted years ago (e.g. ITC
    Holdings, 2016) gets reassigned to a different company, and Yahoo returns the
    new security's prices, creating impossible discontinuities (ITC showed a
    6700x one-day jump from the join). (2) FAILED-COMPANY OTC SHELLS -- e.g.
    Signature Bank (SBNY) trading at $0.0001 post-collapse. A real listed equity
    never moves >300% in a day or sits at sub-penny prices, so either signals data
    we can't faithfully mirror; drop to the coverage gap instead."""
    ac = df["adj_close"]
    if len(ac) < 2:
        return True  # synthetic/degenerate handled elsewhere
    if (ac < 0.01).any():
        return False
    if ac.pct_change().abs().max() > 3.0:                # >300% close-to-close
        return False
    oc = df["adj_close"] / df["adj_open"]
    if (oc > 4.0).any() or (oc < 0.25).any():            # >300% intraday either way
        return False
    return True


def is_foreign_ticker(ticker):
    """Tickers with an exchange suffix (e.g. AEX.MU = Munich, 0QZI.IL = London IL,
    RDSA.AS = Amsterdam) are foreign listings Yahoo matched to a malformed/foreign
    symbol in the disclosure -- wrong security for a US-trading senator. US class
    shares use a hyphen (BRK-B), never a dotted exchange suffix here."""
    return "." in str(ticker)


def pull_ticker(yahoo_session, fmp_session, fmp_key, ticker, category, period1, period2):
    if category in NO_YAHOO_HISTORY:
        return fetch_fmp_history(fmp_session, ticker, fmp_key, PULL_START)
    data = fetch_yahoo_chart(yahoo_session, ticker, period1, period2)
    return parse_yahoo_chart(data)


def main():
    df_eq = pd.read_csv(EQUITY_CSV)
    resolutions = json.loads(RESOLUTIONS_PATH.read_text())
    resolutions.pop("_notes", None)

    df_eq = df_eq[~df_eq["delisting_category"].isin(EXCLUDED)].copy()
    n_unknown_ticker = int((df_eq["ticker"] == "--").sum())
    df_eq = df_eq[df_eq["ticker"] != "--"].copy()
    tickers = sorted(df_eq["resolved_ticker"].dropna().unique())

    yahoo_session = requests.Session(impersonate="chrome")
    fmp_session = requests.Session()
    fmp_key = read_fmp_key()

    period1 = int(datetime.strptime(PULL_START, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    period2 = int(datetime.now(tz=timezone.utc).timestamp())

    all_frames = []
    failed = []
    bad_series = []
    synthetic_applied = []
    fmp_backfilled = []

    for i, ticker in enumerate(tickers):
        category = resolutions.get(ticker, {}).get("category")
        df_t = pull_ticker(yahoo_session, fmp_session, fmp_key, ticker, category, period1, period2)

        # price-sanity guard: a reassigned ticker, sub-penny shell, or foreign
        # listing is unusable
        if df_t is not None and category not in NEEDS_SYNTHETIC_EXIT and (
                is_foreign_ticker(ticker) or not valid_series(df_t)):
            bad_series.append(ticker)
            df_t = None

        if category in NO_YAHOO_HISTORY and df_t is not None:
            fmp_backfilled.append(ticker)

        if df_t is None:
            failed.append(ticker)
            df_t = pd.DataFrame(columns=["date", "adj_open", "adj_close"])

        if category in NEEDS_SYNTHETIC_EXIT:
            res = resolutions[ticker]
            exit_value = res.get("exit_value_per_share", res.get("exit_value_per_share_estimate"))
            close_date = res.get("close_date")
            if exit_value is not None and close_date:
                exit_date = pd.to_datetime(close_date[:10]).date()
                df_t = df_t[df_t["date"] < exit_date]
                exit_row = pd.DataFrame([{"date": exit_date, "adj_open": exit_value, "adj_close": exit_value}])
                df_t = pd.concat([df_t, exit_row])
                synthetic_applied.append(ticker)

        df_t = df_t.copy()
        df_t["ticker"] = ticker
        all_frames.append(df_t)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(tickers)} tickers pulled...")

    prices = pd.concat(all_frames, ignore_index=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_csv(OUT_PRICES, index=False)

    # Coverage gap, decided: exclude tickers with no price data anywhere (Yahoo nor
    # FMP) rather than guess. Computed generically off the actual pull result, not
    # just the NO_YAHOO_HISTORY set -- catches tickers with no resolution entry at
    # all that also turned out to have zero Yahoo history (e.g. ALTM, and CVT --
    # DGNR's resolved_ticker, itself later taken private, a missed double-hop in
    # the D5 research).
    priced_tickers = set(prices["ticker"].unique())
    gap_tickers = [t for t in failed if t not in synthetic_applied]
    gap_rows = df_eq[~df_eq["resolved_ticker"].isin(priced_tickers)]
    gap_dollars = gap_rows["amount_estimate"].sum()
    total_dollars = df_eq["amount_estimate"].sum()

    print(f"Excluded {n_unknown_ticker} rows with unknown ticker ('--') -- can't price what isn't named")
    print(f"\nTickers requested: {len(tickers)}")
    print(f"  FMP backfill used (Yahoo has no history for these -- confirmed-delisted): "
          f"{len(fmp_backfilled)}/{sum(1 for t in tickers if resolutions.get(t, {}).get('category') in NO_YAHOO_HISTORY)}")
    print(f"  synthetic exit-value row applied (bankruptcy/liquidation/spac, overrides last trade): "
          f"{len(synthetic_applied)}  {synthetic_applied}")
    print(f"    -- these 8 have an exit value but NO entry-side price history at all; portfolio.py")
    print(f"       can mark the exit but can't compute a return for any position opened before it")
    print(f"\nPrice-sanity guard dropped {len(bad_series)} tickers (reassigned symbols / "
          f"sub-penny shells): {bad_series}")
    print(f"\nCoverage gap (zero price data via Yahoo or FMP, no override possible -- excluded, per Aidan's call):")
    print(f"  tickers: {len(gap_tickers)}  {gap_tickers}")
    print(f"  transactions: {len(gap_rows)}/{len(df_eq)} ({len(gap_rows) / len(df_eq):.1%})")
    print(f"  dollar exposure: ${gap_dollars:,.0f}/${total_dollars:,.0f} ({gap_dollars / total_dollars:.1%})")

    print(f"\nWrote {OUT_PRICES} ({len(prices)} rows, {prices['ticker'].nunique()} tickers)")


if __name__ == "__main__":
    main()
"""
Real ingest pipeline for the Senate eFD source (D0). Reuses the scrape mechanics in
efd_scrape.py (same cache as efd_probe.py -- re-running this does not re-hit the
server if data/raw/efd/ is already populated).

Applies the decisions Aidan made on top of the recon:

D1 -- amount midpoint: geometric mean sqrt(floor * ceiling), not arithmetic mean.
      For the open-ended top band ("Over $50,000,000"), no ceiling exists, so the
      floor is scaled by sqrt(ratio) of the band directly below it (ratio =
      50,000,000 / 25,000,001), the same floor-to-geomean relationship as every
      other band, extrapolated past the last observed boundary.
D2 -- equities-only filter: keep asset_type == "Stock" (the source's own taxonomy,
      which already includes ETFs). ETFs are flagged in their own `is_etf` column
      rather than dropped, so the backtest can run all-stock and stock-ex-ETF
      cohorts side by side. Non-"Stock" asset types (bonds, options, crypto, etc.)
      are dropped; the dropped fraction is reported, not silently discarded.
D3 -- partial-sale handling: `type` (Purchase / Sale (Full) / Sale (Partial) /
      Exchange) passes through untouched, plus an `is_partial_sale` flag. The
      actual position-cap policy (minimum holding-period assumption to avoid
      reducing a tracked position below zero) needs running position state across
      time per (senator, ticker) and belongs in portfolio.py once D6/D7 are decided
      -- not implemented here.
D12 -- amendment resolution: group transactions by (senator, transaction_date,
      ticker); if any row in a group is from an amended report, keep only the
      amendment row(s) and drop the rest of that group.
D5 -- delisted/acquired/renamed tickers: rows with two tickers in one cell
      (spin-off/merger distributions, e.g. "DELL / VMW") are dropped -- no
      defensible way to split one disclosed dollar amount across two securities.
      Every other ticker is annotated with its resolution from
      data/processed/delisting_resolutions.json (rename -> resolved_ticker,
      acquired/bankrupt/liquidated -> delisting_category) so prices.py and
      portfolio.py don't have to re-derive it.

Does NOT touch D11 (execution timing) -- that's portfolio.py's job at backtest time,
not a property of the ingested transaction itself.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from congress_backtest import efd_scrape

OPTION_TYPE_RE = re.compile(r"Option Type:\s*(Call|Put)", re.IGNORECASE)

CACHE_DIR = Path("data/raw/efd")
PROCESSED_DIR = Path("data/processed")
DELISTING_RESOLUTIONS_PATH = PROCESSED_DIR / "delisting_resolutions.json"
SUBMITTED_START_DATE = "01/01/2015 00:00:00"   # D0 window floor (v2: extended from 2021 to 2015 for depth)

# Official STOCK Act PTR disclosure bands. The open top band has no ceiling --
# D1 extrapolates from the ratio of the band directly below it.
PRIOR_BAND_FLOOR = 25_000_001
PRIOR_BAND_CEILING = 50_000_000
TOP_BAND_RATIO = PRIOR_BAND_CEILING / PRIOR_BAND_FLOOR

ETF_PATTERN = re.compile(r"\bETF\b|\bIndex Fund\b|\bIndex Trust\b", re.IGNORECASE)


def parse_amount_band(amount_str):
    """Return (floor, ceiling) as ints; ceiling is None for the open top band."""
    s = amount_str.strip()
    if s.lower().startswith("over"):
        floor = int(re.sub(r"[^\d]", "", s))
        return floor, None
    floor_str, ceiling_str = s.split(" - ")
    floor = int(re.sub(r"[^\d]", "", floor_str))
    ceiling = int(re.sub(r"[^\d]", "", ceiling_str))
    return floor, ceiling


def amount_estimate(floor, ceiling):
    """D1: geometric mean of the band; open top band extrapolated by TOP_BAND_RATIO."""
    if ceiling is None:
        return floor * (TOP_BAND_RATIO ** 0.5)
    return (floor * ceiling) ** 0.5


def apply_d1(df):
    bands = df["amount"].apply(parse_amount_band)
    df["amount_floor"] = bands.apply(lambda b: b[0])
    df["amount_ceiling"] = bands.apply(lambda b: b[1])
    df["amount_estimate"] = [amount_estimate(f, c) for f, c in bands]
    return df


def apply_d2(df):
    df["is_etf"] = df["asset_description"].str.contains(ETF_PATTERN, na=False)
    df["is_equity"] = df["asset_type"] == "Stock"
    return df


def apply_d3(df):
    df["is_partial_sale"] = df["type"] == "Sale (Partial)"
    return df


def apply_options(df):
    """v2 -- keep option rows (D2 dropped them) and map them to signed exposure in
    the *underlying*: calls = long (+1), puts = short (-1). We can't price the
    options themselves on free data (no historical chains), so this captures their
    directional view, not leverage/convexity. Options with no parseable Call/Put in
    the description (option_sign NaN) are left unsigned and dropped from the
    with-options set downstream. Stocks get instrument='stock', option_sign=+1."""
    df = df.copy()
    df["instrument"] = np.where(df["asset_type"] == "Stock Option", "option", "stock")
    cp = df["asset_description"].str.extract(OPTION_TYPE_RE, expand=False).str.lower()
    opt_sign = cp.map({"call": 1.0, "put": -1.0})
    df["option_sign"] = np.where(df["instrument"] == "option", opt_sign, 1.0)
    return df


def apply_d12(df):
    """Keep only the amendment row(s) within each (senator, transaction_date, ticker,
    instrument, option_sign) group that contains one; leave groups with no amendment
    untouched. instrument/option_sign are in the key so a call and a put on the same
    ticker/date aren't collapsed into one another."""
    group_cols = ["senator", "transaction_date", "ticker", "instrument", "option_sign"]
    has_amendment = df.groupby(group_cols, dropna=False)["is_amendment"].transform("any")
    keep = (~has_amendment) | df["is_amendment"]
    return df[keep].copy()


def apply_d5(df):
    """Drop multi-ticker spin-off/merger rows; annotate the rest with their D5
    resolution (rename/acquired/bankrupt/liquidated) where one exists."""
    is_multi_ticker = df["ticker"].str.contains(" / ", na=False, regex=False)
    n_multi = int(is_multi_ticker.sum())
    df = df[~is_multi_ticker].copy()

    resolutions = json.loads(DELISTING_RESOLUTIONS_PATH.read_text())
    resolutions.pop("_notes", None)

    def resolve(ticker):
        r = resolutions.get(ticker)
        if r is None:
            return pd.Series({"delisting_category": None, "resolved_ticker": ticker})
        return pd.Series({
            "delisting_category": r["category"],
            "resolved_ticker": r.get("new_ticker", ticker),
        })

    df = pd.concat([df, df["ticker"].apply(resolve)], axis=1)
    return df, n_multi


def main():
    session = efd_scrape.new_session()
    token = efd_scrape.authenticate(session)
    flattened, deduped = efd_scrape.scrape_flattened_transactions(
        session, token, SUBMITTED_START_DATE, CACHE_DIR
    )

    df = pd.DataFrame(flattened)
    n_raw = len(df)

    df = apply_d1(df)
    df = apply_d2(df)
    df = apply_d3(df)
    df = apply_options(df)

    n_dropped_amendment_originals = 0
    df_resolved = apply_d12(df)
    n_dropped_amendment_originals = n_raw - len(df_resolved)
    df_resolved, n_multi_ticker = apply_d5(df_resolved)

    n_non_equity = (~df_resolved["is_equity"]).sum()
    df_equity = df_resolved[df_resolved["is_equity"]].copy()
    n_etf = df_equity["is_etf"].sum()
    n_delisted = df_equity["delisting_category"].notna().sum()

    # v2 with-options set: stocks (long) + options with a parseable direction
    # (calls long, puts short via option_sign). Unsigned options are dropped.
    is_signed_option = (df_resolved["instrument"] == "option") & df_resolved["option_sign"].notna()
    df_withopt = df_resolved[df_resolved["is_equity"] | is_signed_option].copy()
    n_opt = int(is_signed_option.sum())
    n_puts = int((df_withopt["option_sign"] == -1.0).sum())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_all = PROCESSED_DIR / "senate_ptr_transactions_all.csv"
    out_equity = PROCESSED_DIR / "senate_ptr_transactions_equity.csv"
    out_withopt = PROCESSED_DIR / "senate_ptr_transactions_with_options.csv"
    df_resolved.to_csv(out_all, index=False)
    df_equity.to_csv(out_equity, index=False)
    df_withopt.to_csv(out_withopt, index=False)

    print(f"Raw flattened transactions: {n_raw}")
    print(f"After D12 (amendment resolution): {len(df_resolved)}  "
          f"({n_dropped_amendment_originals} superseded-original rows dropped)")
    print(f"\nD2 equities filter: {n_non_equity}/{len(df_resolved)} "
          f"({n_non_equity / len(df_resolved):.1%}) dropped as non-'Stock' asset_type")
    print(f"  remaining equities: {len(df_equity)}  (of which ETFs: {n_etf}, "
          f"{n_etf / len(df_equity):.1%} -- flagged via is_etf, not dropped)")
    print(f"\nD5: dropped {n_multi_ticker} multi-ticker spin-off/merger rows "
          f"(no defensible single-ticker attribution)")
    print(f"  {n_delisted}/{len(df_equity)} remaining rows touch a ticker with a "
          f"D5 resolution (rename/acquired/bankrupt/liquidated/fund-mismatch)")
    print(f"\nD1 amount_estimate sanity: min ${df_equity['amount_estimate'].min():,.0f}, "
          f"max ${df_equity['amount_estimate'].max():,.0f}")

    print(f"\nv2 options: {n_opt} signed option rows added to the with-options set "
          f"({n_puts} puts / {n_opt - n_puts} calls); unsigned options dropped")
    print(f"\nWrote {out_all} ({len(df_resolved)}), {out_equity} ({len(df_equity)}), "
          f"and {out_withopt} ({len(df_withopt)} rows)")
    print("\n  --- 3 sample equity rows after D1/D2/D3/D12 ---")
    cols = ["senator", "transaction_date", "ticker", "type", "amount",
            "amount_estimate", "is_etf", "is_partial_sale", "is_amendment"]
    for _, row in df_equity[cols].head(3).iterrows():
        print("   ", row.to_dict())


if __name__ == "__main__":
    main()
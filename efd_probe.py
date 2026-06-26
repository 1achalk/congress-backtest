"""
efd_probe.py - recon scrape of the official Senate eFD portal, NOT part of the backtest.

Scope: Senate PTRs only, submitted_start_date 2021-01-01 -> present. Answers, before
building ingest.py for real:
  1. How many PTRs are in the window, and what fraction are paper (uninspectable for v1)?
  2. What does a flattened transaction row actually look like?
  3. How big is the unknown-ticker fraction, and how many filings are amendments?

Does NOT apply D1 (midpoint), D2 (equities filter), D3 (partial sales), D11 (execution
timing), or D12 (amendment/duplicate policy) -- raw fields pass through untouched.
Those are decided now (see CLAUDE.md) and applied for real in congress_backtest/ingest.py,
which reuses the same scrape mechanics from congress_backtest/efd_scrape.py.

NOTE: confirm every endpoint/field against the LIVE site before reusing this -- gov
sites drift. Confirmed 2026-06-21, see congress_backtest/efd_scrape.py docstring.
"""

import json
from datetime import datetime
from pathlib import Path

from congress_backtest import efd_scrape

CACHE_DIR = Path("data/raw/efd")
SUBMITTED_START_DATE = "01/01/2015 00:00:00"   # D0 window floor (v2: extended from 2021 to 2015 for depth)


def parse_date(s, fmt="%m/%d/%Y"):
    return datetime.strptime(s, fmt)


def main():
    session = efd_scrape.new_session()
    token = efd_scrape.authenticate(session)

    flattened, deduped = efd_scrape.scrape_flattened_transactions(
        session, token, SUBMITTED_START_DATE, CACHE_DIR
    )

    n_paper = sum(1 for r in deduped if r["is_paper"])
    n_electronic = len(deduped) - n_paper
    n_amendments = sum(1 for r in deduped if r["is_amendment"])

    out_path = efd_scrape.cache_path(CACHE_DIR, "flattened_transactions_recon.json")
    out_path.write_text(json.dumps(flattened, indent=2))

    print(f"\nPTR reports in window: {len(deduped)} unique")
    print(f"  electronic: {n_electronic}   paper: {n_paper}   (paper coverage gap: {n_paper / len(deduped):.1%})")
    print(f"  flagged as amendments (title contains 'Amendment'): {n_amendments}  -- not deduped, see D12")

    print(f"\nFlattened transactions (electronic reports only): {len(flattened)}")
    if flattened:
        tx_dates = sorted((parse_date(t["transaction_date"]), t["transaction_date"]) for t in flattened)
        recv_dates = sorted((parse_date(r["date_received"]), r["date_received"]) for r in deduped)
        unknown_ticker = sum(1 for t in flattened if t["ticker"].strip() in ("--", ""))
        print(f"  transaction_date range: {tx_dates[0][1]} -> {tx_dates[-1][1]}")
        print(f"  date_received range:    {recv_dates[0][1]} -> {recv_dates[-1][1]}")
        print(f"  unknown/'--' ticker fraction: {unknown_ticker}/{len(flattened)} ({unknown_ticker / len(flattened):.1%})")
        print("\n  --- 3 sample flattened rows ---")
        for t in flattened[:3]:
            print("   ", t)

    print(f"\nFull flattened set cached to {out_path}")


if __name__ == "__main__":
    main()
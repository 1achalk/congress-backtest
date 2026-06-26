"""
Portfolio construction: turns the cleaned congressional signals into a daily
return series for a given cohort and entry-date basis (D6, D7, D11, and the
deferred-D3 holding rule). This is the core module.

Construction (all settled with Aidan):

D6  -- fixed-AUM, scale the book. There is one constant notional capital base.
       Target weight on each ticker is proportional to the cohort's *net held*
       disclosed midpoint in that ticker; weights are renormalised to sum to 1
       whenever anything is held. New signals don't add capital on top -- they
       re-slice the same pie (funding a new name trims the others).
D7  -- idle cash earns the risk-free rate (FRED 3-month T-bill, DTB3). Under a
       scale-the-book construction the book is always 100% invested *whenever it
       holds anything*, so idle cash only appears in flat periods (before the
       first signal, or any stretch where net held positions net to zero). Those
       days earn the daily T-bill rate.
D11 -- both legs fill at the NEXT session's open after the relevant date
       (transaction_date for the trade-date basis, date_received for the
       disclosure-date basis), same rule for both so their gap measures only
       reporting lag. A name entering on day t books that day's open->close move
       (adj_close/adj_open - 1); everything else is close-to-close.
Holding rule (deferred D3) -- hold until disclosed sale. Net held midpoint per
       ticker walks max(0, running + delta): Purchase +midpoint, Sale (Full) and
       Sale (Partial) both -midpoint, floored at zero so it can never go short.
       A name with no disclosed sale is held until its price series ends (or to
       the window end). Exchange (7 rows) is treated as a no-op for sizing.

Aggregate-netting note: net held is summed across the whole cohort per ticker, not
tracked per senator. A "Sale (Full)" therefore subtracts only its own disclosed
midpoint (not the seller's true lot), which is the simple, position-state-free
reading of "hold until disclosed sale." Per-senator lot tracking would be a
refinement, not v1.

Forced-liquidation note: when a held ticker's price series ends (a synthetic
bankruptcy/exit mark from D5, or just end of data), it is valued through its last
available price and then drops out; the scale-the-book renormalisation spreads its
capital across the surviving names the next day.

NOT here: D8 (transaction costs), D9 (top-N cohort selection / train-test split),
D10 (SPY benchmark). This module produces one cohort's gross return series; the
four-way table and benchmark assembly live in backtest.py once those land.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
from curl_cffi import requests

PROCESSED_DIR = Path("data/processed")
EQUITY_CSV = PROCESSED_DIR / "senate_ptr_transactions_equity.csv"
WITH_OPTIONS_CSV = PROCESSED_DIR / "senate_ptr_transactions_with_options.csv"
PRICES_CSV = PROCESSED_DIR / "prices.csv"
RF_CACHE = Path("data/raw/fred/DTB3.csv")

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
TRADING_DAYS_PER_YEAR = 252

SIGNED = {"Purchase": 1.0, "Sale (Full)": -1.0, "Sale (Partial)": -1.0, "Exchange": 0.0}


def load_prices():
    df = pd.read_csv(PRICES_CSV, parse_dates=["date"])
    adj_close = df.pivot_table(index="date", columns="ticker", values="adj_close")
    adj_open = df.pivot_table(index="date", columns="ticker", values="adj_open")
    adj_close = adj_close.sort_index()
    adj_open = adj_open.reindex_like(adj_close)
    return adj_close, adj_open


def fetch_risk_free(calendar):
    """Daily risk-free return aligned to `calendar` (DTB3 is annualised percent)."""
    RF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if RF_CACHE.exists():
        raw = RF_CACHE.read_text()
    else:
        r = requests.get(FRED_URL, params={"id": "DTB3"}, impersonate="chrome", timeout=20)
        r.raise_for_status()
        raw = r.text
        RF_CACHE.write_text(raw)
    s = pd.read_csv(io.StringIO(raw), parse_dates=["observation_date"])
    s = s.rename(columns={"observation_date": "date", "DTB3": "rate"})
    s["rate"] = pd.to_numeric(s["rate"], errors="coerce")  # FRED uses "." for missing
    s = s.set_index("date")["rate"].sort_index()
    daily = s.reindex(calendar.union(s.index)).ffill().reindex(calendar)
    daily = daily.fillna(0.0) / 100.0 / TRADING_DAYS_PER_YEAR
    return daily


def build_held_matrix(tx, cohort, entry_basis, calendar, valid_tickers, floor=True):
    """Daily net-held-exposure matrix (dates x tickers) for one cohort/basis.

    floor=True: held = max(0, running + delta) -- the long-only stock rule (a
    disclosed sale can't drive the position short). floor=False: held = running +
    delta, signed -- used for the option layer so puts (option_sign=-1) create
    short exposure in the underlying (v2). If an `option_sign` column is present it
    multiplies the signed delta (calls +1 / puts -1; stocks +1)."""
    df = tx.copy()
    if cohort != "all":
        df = df[df["senator"].isin(cohort)]
    df = df[df["resolved_ticker"].isin(valid_tickers)].copy()
    if df.empty:
        return pd.DataFrame(0.0, index=calendar, columns=[])

    basis_col = "transaction_date" if entry_basis == "trade" else "date_received"
    df["basis_date"] = pd.to_datetime(df[basis_col], errors="coerce")
    df = df.dropna(subset=["basis_date"])
    sign = df["option_sign"] if "option_sign" in df.columns else 1.0
    df["signed_midpoint"] = df["type"].map(SIGNED).fillna(0.0) * df["amount_estimate"] * sign

    # D11: fill at the next trading day strictly after the basis date.
    cal = calendar.sort_values()
    pos = cal.searchsorted(df["basis_date"].values, side="right")
    in_range = pos < len(cal)
    df = df[in_range].copy()
    df["fill_date"] = cal[pos[in_range]]

    # per-ticker: walk events, held = (max(0, .) if floor else .), record at fills
    held = pd.DataFrame(0.0, index=cal, columns=sorted(df["resolved_ticker"].unique()))
    deltas = df.groupby(["fill_date", "resolved_ticker"])["signed_midpoint"].sum()
    for ticker, tk_deltas in deltas.groupby(level="resolved_ticker"):
        running = 0.0
        series = {}
        for (fill_date, _), delta in tk_deltas.items():
            running = max(0.0, running + delta) if floor else running + delta
            series[fill_date] = running
        col = pd.Series(series).reindex(cal).ffill().fillna(0.0)
        held[ticker] = col
    return held


def load_data():
    """Load everything the engine needs once, so callers that run it many times
    (e.g. backtest.py ranking ~20 senators x 2 bases) don't reload per call."""
    adj_close, adj_open = load_prices()
    rf = fetch_risk_free(adj_close.index)
    tx = pd.read_csv(EQUITY_CSV)
    tx_withopt = pd.read_csv(WITH_OPTIONS_CSV) if WITH_OPTIONS_CSV.exists() else tx
    return {"adj_close": adj_close, "adj_open": adj_open, "rf": rf,
            "tx": tx, "tx_withopt": tx_withopt}


def cohort_held(data, cohort, entry_basis, calendar, valid_tickers, include_options):
    """Combined held-exposure matrix: long-only stock layer, plus (optionally) a
    signed option layer (calls long, puts short) summed on top."""
    if not include_options:
        return build_held_matrix(data["tx"], cohort, entry_basis, calendar, valid_tickers, floor=True)
    tx = data["tx_withopt"]
    stock = build_held_matrix(tx[tx["instrument"] == "stock"], cohort, entry_basis,
                              calendar, valid_tickers, floor=True)
    opt = build_held_matrix(tx[tx["instrument"] == "option"], cohort, entry_basis,
                            calendar, valid_tickers, floor=False)
    return stock.add(opt, fill_value=0.0).fillna(0.0)


def run_portfolio(cohort="all", entry_basis="trade", cost_bps=0.0, data=None,
                  include_options=False):
    """Daily DataFrame: port_ret (net of costs), equity_curve, n_positions,
    cash_weight, turnover. cost_bps (D8) is charged on each buy and sell, i.e. on
    total notional traded = sum|delta weight|, every rebalance day. With
    include_options=True the book is long-short (calls long, puts short) and is
    normalised to 100% *gross* exposure (sum|weight|); long-only it reduces to the
    original net-long book."""
    if data is None:
        data = load_data()
    adj_close, adj_open, rf = data["adj_close"], data["adj_open"], data["rf"]
    calendar = adj_close.index
    valid_tickers = set(adj_close.columns)

    held = cohort_held(data, cohort, entry_basis, calendar, valid_tickers, include_options)
    held = held.reindex(columns=[c for c in held.columns if c in valid_tickers])

    # only hold names that actually have a price that day; renormalise over them.
    # Normalise by GROSS exposure (sum|held|) so long-short books cap at 100% gross
    # and long-only books are unchanged (held >= 0 -> sum|held| == sum held).
    priced = adj_close[held.columns].notna()
    held_priced = held.where(priced, 0.0)
    gross = held_priced.abs().sum(axis=1)
    invested = gross > 0
    weights = held_priced.div(gross.where(invested), axis=0).fillna(0.0)

    # per-name daily return; names entering at today's open book open->close
    close = adj_close[held.columns]
    open_ = adj_open[held.columns]
    r_close = close / close.shift(1) - 1.0
    r_openentry = close / open_ - 1.0
    entered_today = (weights.abs() > 0) & (weights.shift(1).fillna(0.0).abs() == 0)
    r_cell = r_close.where(~entered_today, r_openentry)
    # Backstop: cap per-name daily returns at +/-50%. A real US equity essentially
    # never moves that much close-to-close; anything beyond is a residual data
    # glitch (e.g. a corporate-action join Yahoo hasn't smoothed, like DuPont's
    # 2026 reverse split) that the prices.py sanity guard didn't already drop. This
    # keeps one bad print from blowing up a senator's compounded return.
    r_cell = r_cell.clip(-0.5, 0.5)

    invested_ret = (weights * r_cell.fillna(0.0)).sum(axis=1)
    gross_ret = invested_ret.where(invested, rf)

    # D8 turnover cost: drift yesterday's weights by realised returns, compare to
    # today's target, charge cost_bps on the absolute weight change (buys + sells).
    w_prev = weights.shift(1).fillna(0.0)
    drifted = w_prev * (1.0 + r_cell.fillna(0.0))
    drifted_gross = drifted.abs().sum(axis=1)
    drifted = drifted.div(drifted_gross.where(drifted_gross > 0), axis=0).fillna(0.0)
    turnover = (weights - drifted).abs().sum(axis=1)
    cost = (cost_bps / 1e4) * turnover
    port_ret = gross_ret - cost

    # start the series at the first day anything is actually held
    first = invested.idxmax() if invested.any() else calendar[0]
    port_ret = port_ret.loc[first:]
    out = pd.DataFrame({
        "port_ret": port_ret,
        "equity_curve": (1.0 + port_ret).cumprod(),
        "n_positions": (weights.loc[first:].abs() > 0).sum(axis=1),
        "cash_weight": (~invested.loc[first:]).astype(float),
        "turnover": turnover.loc[first:],
    })
    return out


def main():
    data = load_data()
    for basis in ("trade", "disclosure"):
        out = run_portfolio(cohort="all", entry_basis=basis, data=data)
        total = out["equity_curve"].iloc[-1] - 1.0
        yrs = (out.index[-1] - out.index[0]).days / 365.25
        cagr = out["equity_curve"].iloc[-1] ** (1 / yrs) - 1.0
        ann_vol = out["port_ret"].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        print(f"\n[all-Congress, {basis}-date entry]")
        print(f"  window: {out.index[0].date()} -> {out.index[-1].date()} ({yrs:.1f}y)")
        print(f"  total return: {total:.1%}   CAGR: {cagr:.1%}   ann vol: {ann_vol:.1%}")
        print(f"  max positions held: {out['n_positions'].max()}   "
              f"days flat (cash): {int(out['cash_weight'].sum())}")


if __name__ == "__main__":
    main()
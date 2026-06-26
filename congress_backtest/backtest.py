"""
Four-way backtest assembly + honesty gaps (D8, D9, D10).

Structure (no look-ahead):
  - Common evaluable window = [common_start, common_end], where common_start is the
    later of the two bases' first-fill dates (so trade and disclosure are aligned)
    and common_end is the last price date.
  - 60/40 chronological split (D9): TRAIN = first 60% of the window, TEST = last 40%.
  - D9 top-N: rank every senator with >= MIN_TRAIN_TRADES disclosed equity trades
    in the train window by their own single-senator mirror *total return over the
    train window* (gross), and take the top 5. Selection uses train data only; the
    frozen top-5 set is then evaluated on the held-out test window.
    Ranking is done on the DISCLOSURE basis only, for two reasons: (1) it's the
    only performance an investor can actually observe in real time -- ranking on
    trade-date-timed returns would use information you never cleanly see, a subtle
    look-ahead in the *selection* step; (2) using one frozen set for both bases
    keeps both honesty gaps clean -- the reporting-lag gap then compares identical
    cohorts across timings, and the selection gap compares all-vs-top5 within a
    timing. (Per-basis ranking conflated the two and flipped the top-5 lag gap.)
  - The four cells {all-Congress, top-5} x {trade, disclosure} are each evaluated on
    the *same* test window, gross (D8 = 0 bps) and net (D8 = 5 bps per side).
  - D10: SPY buy-and-hold total return over the identical test window.

Honesty gaps (the headline deliverables):
  - reporting-lag gap  = trade-date minus disclosure-date return (per cohort)
  - selection gap      = top-5 minus all-Congress return (per basis)

Metrics come from congress_backtest.metrics (a placeholder stand-in for Aidan's
Sharpe/Sortino/MDD lib until it's wired in).
"""

import json
from pathlib import Path

import pandas as pd
from curl_cffi import requests

from congress_backtest import metrics, portfolio

PROCESSED_DIR = Path("data/processed")
SPY_CACHE = Path("data/raw/yahoo/SPY.json")
OUT_TABLE = PROCESSED_DIR / "backtest_fourway.csv"

TRAIN_FRAC = 0.60
TOP_N = 5
MIN_TRAIN_TRADES = 10
COST_BPS = 5.0
BASES = ("trade", "disclosure")


def first_fill_date(data, basis):
    out = portfolio.run_portfolio(cohort="all", entry_basis=basis, data=data)
    return out.index[0]


def define_windows(data):
    common_start = max(first_fill_date(data, b) for b in BASES)
    common_end = data["adj_close"].index[-1]
    cal = data["adj_close"].index
    span_days = (common_end - common_start).days
    split_target = common_start + pd.Timedelta(days=int(TRAIN_FRAC * span_days))
    split = cal[cal.searchsorted(split_target)]
    return common_start, split, common_end


def train_total_return(port_ret, train_start, split):
    seg = port_ret.loc[train_start:split]
    return (1.0 + seg).prod() - 1.0


def rank_top_n(data, basis, train_start, split):
    tx = data["tx"]
    basis_col = "transaction_date" if basis == "trade" else "date_received"
    bd = pd.to_datetime(tx[basis_col], errors="coerce")
    in_train = (bd >= train_start) & (bd <= split)
    train_counts = tx[in_train].groupby("senator").size()
    rankable = train_counts[train_counts >= MIN_TRAIN_TRADES].index.tolist()

    scores = {}
    for senator in rankable:
        out = portfolio.run_portfolio(cohort=[senator], entry_basis=basis, data=data)
        scores[senator] = train_total_return(out["port_ret"], train_start, split)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [s for s, _ in ranked[:TOP_N]], ranked


def fetch_spy(calendar):
    if SPY_CACHE.exists():
        data = json.loads(SPY_CACHE.read_text())
    else:
        p1 = int(pd.Timestamp("2014-01-01", tz="UTC").timestamp())
        p2 = int(pd.Timestamp.now(tz="UTC").timestamp())
        s = requests.Session(impersonate="chrome")
        r = s.get("https://query1.finance.yahoo.com/v8/finance/chart/SPY",
                  params={"period1": p1, "period2": p2, "interval": "1d", "events": "div,splits"},
                  timeout=20)
        data = r.json()
        SPY_CACHE.write_text(json.dumps(data))
    res = data["chart"]["result"][0]
    ts = res["timestamp"]
    adj = res["indicators"]["adjclose"][0]["adjclose"]
    dates = pd.to_datetime([pd.Timestamp(t, unit="s").normalize() for t in ts])
    spy = pd.Series(adj, index=dates).dropna()
    spy.index = spy.index.tz_localize(None)
    return spy.reindex(calendar).ffill()


def stats(daily_ret, daily_rf):
    return {
        "total_return": (1.0 + daily_ret).prod() - 1.0,
        "cagr": metrics.annualized_return(daily_ret),
        "ann_vol": metrics.annualized_vol(daily_ret),
        "sharpe": metrics.sharpe(daily_ret, daily_rf),
        "sortino": metrics.sortino(daily_ret, daily_rf),
        "max_dd": metrics.max_drawdown(daily_ret),
    }


def main():
    data = portfolio.load_data()
    common_start, split, common_end = define_windows(data)
    rf = data["rf"]
    rf_test = rf.loc[split:common_end]

    print(f"Common window: {common_start.date()} -> {common_end.date()}")
    print(f"  TRAIN (rank): {common_start.date()} -> {split.date()}")
    print(f"  TEST  (eval): {split.date()} -> {common_end.date()}")

    # D9: rank ONCE on the observable (disclosure) basis; freeze for both bases.
    top_set, ranked = rank_top_n(data, "disclosure", common_start, split)
    print(f"\nTop {TOP_N} by train-window own-mirror return [disclosure basis, "
          f"frozen for both timings]:")
    for s, score in ranked[:TOP_N]:
        print(f"  {score:+7.1%}  {s}")

    rows = []
    for basis in BASES:
        for cohort_name, cohort in [("all-Congress", "all"), (f"top-{TOP_N}", top_set)]:
            for scope_name, inc_opt in [("stocks", False), ("stocks+options", True)]:
                for cost_name, cost in [("gross", 0.0), (f"net-{int(COST_BPS)}bps", COST_BPS)]:
                    out = portfolio.run_portfolio(cohort=cohort, entry_basis=basis,
                                                  cost_bps=cost, data=data, include_options=inc_opt)
                    ret_test = out["port_ret"].loc[split:common_end]
                    st = stats(ret_test, rf_test)
                    rows.append({"cohort": cohort_name, "basis": basis, "scope": scope_name,
                                 "costs": cost_name, **st})

    spy = fetch_spy(data["adj_close"].index)
    spy_ret = (spy / spy.shift(1) - 1.0).loc[split:common_end]
    rows.append({"cohort": "SPY", "basis": "benchmark", "scope": "benchmark",
                 "costs": "buy&hold", **stats(spy_ret, rf_test)})

    table = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_TABLE, index=False)

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    show = table.copy()
    for c in ["total_return", "cagr", "ann_vol", "max_dd"]:
        show[c] = (show[c] * 100).round(1)
    print("\n=== Full table (test window) ===")
    print(show.to_string(index=False))

    # honesty gaps use the stock-only scope (the primary, long-only construction)
    def net_total(cohort, basis, scope="stocks"):
        m = table[(table.cohort == cohort) & (table.basis == basis)
                  & (table.scope == scope) & (table.costs == f"net-{int(COST_BPS)}bps")]
        return m["total_return"].iloc[0]

    print(f"\n=== Honesty gaps (stocks only, net of {int(COST_BPS)}bps, test total return) ===")
    print("Reporting-lag gap (trade minus disclosure):")
    for cohort in ("all-Congress", f"top-{TOP_N}"):
        print(f"  {cohort:14s}: {net_total(cohort,'trade') - net_total(cohort,'disclosure'):+.1%}")
    print("Selection gap (top-5 minus all-Congress):")
    for basis in BASES:
        print(f"  {basis:14s}: {net_total(f'top-{TOP_N}',basis) - net_total('all-Congress',basis):+.1%}")

    print(f"\n=== Options impact (calls long / puts short, net of {int(COST_BPS)}bps) ===")
    print("with-options minus stocks-only (test total return):")
    for cohort in ("all-Congress", f"top-{TOP_N}"):
        for basis in BASES:
            d = net_total(cohort, basis, "stocks+options") - net_total(cohort, basis, "stocks")
            print(f"  {cohort:14s} {basis:11s}: {d:+.1%}")

    spy_tr = table[table.cohort == "SPY"]["total_return"].iloc[0]
    print(f"\nSPY buy-and-hold (same window): {spy_tr:+.1%}")
    print(f"Wrote {OUT_TABLE}")


if __name__ == "__main__":
    main()
"""
Render the deliverables: the four-way table and three charts.

  - Chart 1 (reporting-lag gap): test-window equity curves, trade-date vs
    disclosure-date entry, all-Congress, with SPY. Isolates reporting lag (same
    cohort, only entry timing differs). Annotated with the bootstrap CI/p.
  - Chart 2 (selection gap): test-window equity curves, top-5 vs all-Congress, on
    the disclosure basis, with SPY. Isolates selection bias (same timing, only the
    cohort differs). Annotated with the bootstrap CI/p.
  - Chart 3 (significance forest plot): every headline gap as an annualized point
    estimate with its stationary-block-bootstrap 95% CI, against a zero line. The
    "nothing clears zero" summary.

Re-runs portfolio.py for the curves and significance.py for the CIs (both cheap off
the cached data).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from congress_backtest import backtest, portfolio, significance

REPORTS_DIR = Path("reports")
TABLE_CSV = backtest.OUT_TABLE


def _rebased(port_ret, start, end):
    seg = port_ret.loc[start:end]
    return (1.0 + seg).cumprod() / (1.0 + seg.iloc[0])  # start at ~1.0


def _annot(gap):
    sig = "***" if gap["p"] < 0.01 else "**" if gap["p"] < 0.05 else "*" if gap["p"] < 0.10 else "ns"
    return (f"bootstrap gap: {gap['obs']:+.1%}/yr\n"
            f"95% CI [{gap['lo']:+.1%}, {gap['hi']:+.1%}]\n"
            f"p = {gap['p']:.2f}  ({sig})")


def main():
    data = portfolio.load_data()
    common_start, split, common_end = backtest.define_windows(data)
    top_set, _ = backtest.rank_top_n(data, "disclosure", common_start, split)
    spy = backtest.fetch_spy(data["adj_close"].index)
    spy_curve = _rebased(spy / spy.shift(1) - 1.0, split, common_end)

    gaps, _ = significance.compute_gaps(data)
    by_name = {g["name"]: g for g in gaps}

    def curve(cohort, basis):
        out = portfolio.run_portfolio(cohort=cohort, entry_basis=basis,
                                      cost_bps=backtest.COST_BPS, data=data)
        return _rebased(out["port_ret"], split, common_end)

    box = dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.6")
    REPORTS_DIR.mkdir(exist_ok=True)

    # Chart 1: reporting-lag gap (all-Congress, trade vs disclosure)
    fig, ax = plt.subplots(figsize=(10, 6))
    curve("all", "trade").plot(ax=ax, label="All-Congress, trade-date entry")
    curve("all", "disclosure").plot(ax=ax, label="All-Congress, disclosure-date entry")
    spy_curve.plot(ax=ax, label="SPY buy & hold", color="black", linestyle="--")
    ax.set_title("Reporting-lag gap (net of 5bps) -- test window\n"
                 "same cohort, only entry timing differs")
    ax.set_ylabel("Growth of $1"); ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    ax.text(0.985, 0.03, _annot(by_name["reporting lag: trade - disclosure"]),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, bbox=box)
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "reporting_lag_gap.png", dpi=120)
    plt.close(fig)

    # Chart 2: selection gap (disclosure basis, top-5 vs all)
    fig, ax = plt.subplots(figsize=(10, 6))
    curve("all", "disclosure").plot(ax=ax, label="All-Congress")
    curve(top_set, "disclosure").plot(ax=ax, label=f"Top-{backtest.TOP_N} (train-selected)")
    spy_curve.plot(ax=ax, label="SPY buy & hold", color="black", linestyle="--")
    ax.set_title("Selection gap (net of 5bps, disclosure-date entry) -- test window\n"
                 "same timing, only the cohort differs")
    ax.set_ylabel("Growth of $1"); ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    ax.text(0.985, 0.03, _annot(by_name["selection: top-5 - all (disclosure)"]),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, bbox=box)
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "selection_gap.png", dpi=120)
    plt.close(fig)

    # Chart 3: significance forest plot -- all gaps with 95% CIs vs zero
    fig, ax = plt.subplots(figsize=(10, 6))
    order = list(reversed(gaps))
    ys = range(len(order))
    for y, g in zip(ys, order):
        sig = g["p"] < 0.05
        color = "tab:red" if sig else "0.4"
        ax.errorbar(g["obs"] * 100, y,
                    xerr=[[(g["obs"] - g["lo"]) * 100], [(g["hi"] - g["obs"]) * 100]],
                    fmt="o", color=color, capsize=4)
        ax.text(g["hi"] * 100 + 1, y, f"p={g['p']:.2f}", va="center", fontsize=8, color="0.3")
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(list(ys)); ax.set_yticklabels([g["name"] for g in order], fontsize=9)
    ax.set_xlabel("annualized gap (%/yr), with stationary-block-bootstrap 95% CI")
    ax.set_title("Headline gaps vs zero -- none of the 95% CIs exclude zero\n"
                 "(every gap is statistically indistinguishable from no effect)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "significance_gaps.png", dpi=120)
    plt.close(fig)

    table = pd.read_csv(TABLE_CSV)
    print("Four-way table:\n")
    print(table.to_string(index=False))
    print(f"\nCharts written to {REPORTS_DIR}/: reporting_lag_gap.png, "
          f"selection_gap.png, significance_gaps.png")


if __name__ == "__main__":
    main()
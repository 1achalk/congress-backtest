"""
Bootstrap significance of the headline gaps (v3 add-on).

The reported gaps (selection, vs-SPY, reporting-lag, options) are point estimates
over one ~4.5y out-of-sample path. This asks: are they distinguishable from zero,
or could a concentrated book + one bear market produce them by chance?

Method: stationary block bootstrap (Politis & Romano 1994) on the daily return
DIFFERENCE series d_t = ret_A,t - ret_B,t over the common test window. Daily returns
are autocorrelated and volatility-clustered, so we resample blocks of random
(geometric) length rather than iid days -- an iid bootstrap would understate the
sampling uncertainty and overstate significance. Statistic = mean(d_t), reported
annualized (x252). We report the observed gap, a 95% percentile CI, and a two-sided
percentile p-value for H0: mean daily difference = 0.

Caveat this method shares with the backtest: it treats the price/return data as
given (it does not bootstrap the senator selection or the coverage gaps), so it
bounds sampling noise in the realised series, not model/data-choice uncertainty.
"""

import numpy as np
import pandas as pd

from congress_backtest import backtest, portfolio

B = 10000
EXPECTED_BLOCK = 20      # ~4 trading weeks; >> daily return autocorr horizon
SEED = 12345
COST = backtest.COST_BPS


def stationary_bootstrap_mean(d, b, expected_block, rng):
    """Distribution of the resampled mean of d under the stationary bootstrap."""
    n = len(d)
    d = np.asarray(d)
    p = 1.0 / expected_block
    means = np.empty(b)
    for k in range(b):
        idx = np.empty(n, dtype=np.int64)
        idx[0] = rng.integers(n)
        restart = rng.random(n) < p
        steps = rng.integers(n, size=n)
        for t in range(1, n):
            idx[t] = steps[t] if restart[t] else (idx[t - 1] + 1) % n
        means[k] = d[idx].mean()
    return means


def assess(name, a, b_series, rng):
    """a, b_series: daily-return Series. Test mean(a-b) over their common index.
    Returns a result dict with annualized observed gap, 95% CI, and p-value."""
    d = (a - b_series).dropna()
    obs = d.mean()
    boot = stationary_bootstrap_mean(d.values, B, EXPECTED_BLOCK, rng)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    frac_le0 = (boot <= 0).mean()        # two-sided percentile p for H0: mean = 0
    p = 2 * min(frac_le0, 1 - frac_le0)
    ann = 252
    return {"name": name, "obs": obs * ann, "lo": lo * ann, "hi": hi * ann,
            "p": p, "n": len(d)}


def compute_gaps(data, rng=None):
    """Build the strategy daily-return series and bootstrap every headline gap.
    Returns a list of result dicts (used by both main() and report.py)."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    common_start, split, common_end = backtest.define_windows(data)
    top_set, _ = backtest.rank_top_n(data, "disclosure", common_start, split)

    def series(cohort, basis, include_options):
        out = portfolio.run_portfolio(cohort=cohort, entry_basis=basis,
                                      cost_bps=COST, data=data, include_options=include_options)
        return out["port_ret"].loc[split:common_end]

    spy = backtest.fetch_spy(data["adj_close"].index)
    spy_ret = (spy / spy.shift(1) - 1.0).loc[split:common_end]
    all_disc = series("all", "disclosure", False)
    all_trade = series("all", "trade", False)
    top_disc = series(top_set, "disclosure", False)
    top_trade = series(top_set, "trade", False)
    all_disc_opt = series("all", "disclosure", True)

    gaps = [
        assess("selection: top-5 - all (disclosure)", top_disc, all_disc, rng),
        assess("selection: top-5 - all (trade)", top_trade, all_trade, rng),
        assess("all-Congress - SPY (disclosure)", all_disc, spy_ret, rng),
        assess("top-5 - SPY (disclosure)", top_disc, spy_ret, rng),
        assess("reporting lag: trade - disclosure", all_trade, all_disc, rng),
        assess("options: with-options - stocks", all_disc_opt, all_disc, rng),
    ]
    return gaps, (split, common_end)


def main():
    data = portfolio.load_data()
    gaps, (split, common_end) = compute_gaps(data)
    print(f"Test window {split.date()} -> {common_end.date()}; "
          f"stationary bootstrap B={B}, E[block]={EXPECTED_BLOCK}d")
    print("Gaps = annualized mean daily return differences. *** p<.01 ** p<.05 * p<.10 ns.")
    for g in gaps:
        sig = "***" if g["p"] < 0.01 else "**" if g["p"] < 0.05 else "*" if g["p"] < 0.10 else "ns"
        print(f"\n{g['name']}  ({g['n']} days)")
        print(f"  {g['obs']:+.1%}/yr   95% CI [{g['lo']:+.1%}, {g['hi']:+.1%}]   p={g['p']:.3f}  {sig}")


if __name__ == "__main__":
    main()
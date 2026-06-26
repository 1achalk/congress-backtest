"""
Performance metrics: Sharpe, Sortino, max drawdown.

PLACEHOLDER -- the brief says this module imports Aidan's separate metrics lib and
to "assume it exists." It doesn't exist in this repo yet, and backtest.py needs the
numbers to produce the four-way table, so these are minimal, correct stand-in
implementations. Swap them for Aidan's lib by re-pointing the imports; the
signatures (daily return Series in, scalar out) are what backtest.py depends on.
"""

import numpy as np

TRADING_DAYS_PER_YEAR = 252


def annualized_return(daily_ret):
    eq = (1.0 + daily_ret).prod()
    yrs = len(daily_ret) / TRADING_DAYS_PER_YEAR
    return eq ** (1.0 / yrs) - 1.0 if yrs > 0 else np.nan


def annualized_vol(daily_ret):
    return daily_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe(daily_ret, daily_rf):
    excess = daily_ret - daily_rf
    sd = excess.std()
    return (excess.mean() / sd) * np.sqrt(TRADING_DAYS_PER_YEAR) if sd > 0 else np.nan


def sortino(daily_ret, daily_rf):
    excess = daily_ret - daily_rf
    downside = excess[excess < 0]
    dd = np.sqrt((downside ** 2).mean()) if len(downside) else np.nan
    return (excess.mean() / dd) * np.sqrt(TRADING_DAYS_PER_YEAR) if dd and dd > 0 else np.nan


def max_drawdown(daily_ret):
    eq = (1.0 + daily_ret).cumprod()
    peak = eq.cummax()
    return (eq / peak - 1.0).min()
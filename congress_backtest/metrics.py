# core risk-adjusted performance metrics
# all functions take a pd.Series of daily returns

import numpy as np

def annualized_return(returns, periods_per_year=252):
    """
    Design decisions: README M1 (geometric CAGR off the compounded wealth path).
    """
    G = (1 + returns).prod()
    Y = len(returns)/periods_per_year
    g  = G**(1/Y) - 1
    return g


def annualized_vol(returns, periods_per_year=252):
    """
    Design decisions: README M1 (sqrt-time scaling), M2 (sample std, ddof=1).
    """
    daily_vol = returns.std(ddof=1)
    return daily_vol * (periods_per_year)**(1/2)


def sharpe_ratio(returns, rf_annual=0.0, periods_per_year=252):
    """
    Design decisions: README M1 (annualization), M3 (risk-free rate handling).
    """
    rf_daily = (1+ rf_annual)**(1/periods_per_year) - 1
    excess = returns - rf_daily

    num = excess.mean() * periods_per_year
    denom = annualized_vol(excess, periods_per_year)

    if denom == 0:
        return np.nan


    return num/denom


def sortino_ratio(returns, rf_annual=0.0, periods_per_year=252):
    """
    Design decisions: README M3 (risk-free), M4 (total-N downside deviation).
    """
    rf_daily = (1+ rf_annual)**(1/periods_per_year) - 1
    excess = returns - rf_daily
    
    num = excess.mean() * periods_per_year
    downside = np.minimum(excess, 0)
    msd =  (downside**2).mean()
    denom = np.sqrt(msd) * np.sqrt(periods_per_year)
                                   
    if denom == 0:
        return np.nan

    return num / denom


def max_drawdown(returns):
    """
    Returns dict: {depth, peak_date, trough_date, recovery_date}
    recovery_date is None if wealth never recovers to the prior peak.
    Design decisions: README M5 (drawdown on the compounded wealth path).
    """

    wealth = (1+returns).cumprod()
    running_peak = wealth.cummax()
    draw_down = (wealth - running_peak) / running_peak

    depth = draw_down.min()
    if depth == 0:
        return {'depth': 0.0, 'peak_date': None, 
            'trough_date': None, 'recovery_date': None}

    trough_date = draw_down.idxmin()
    peak_date = wealth[:trough_date].idxmax()
    
    after_trough = wealth[trough_date:]
    recovery = after_trough[after_trough >= wealth[peak_date]]
    if len(recovery) == 0:
        recovery_date = None
    else: 
        recovery_date = recovery.index[0]

    return {"depth": depth, "peak_date": peak_date, 
            "trough_date" : trough_date, "recovery_date": recovery_date}


def rolling_volatility(returns, window=21, periods_per_year=252):
    """
    Returns a Series of rolling annualized volatility.
    Design decisions: README M1 (sqrt-time scaling), M2 (sample std, ddof=1).
    """
    rolling_vol = returns.rolling(window).std(ddof=1)
    return rolling_vol * np.sqrt(periods_per_year)

"""Entry point: run the four-way backtest and render the report.

Assumes the one-time data pulls are already cached (data/raw/). To rebuild those
from scratch, run efd_probe.py, then `python -m congress_backtest.ingest`, then
`python -m congress_backtest.prices`.
"""

from congress_backtest import backtest, report


def main():
    backtest.main()
    report.main()


if __name__ == "__main__":
    main()
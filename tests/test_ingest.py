"""Unit tests for the pure D1 amount-band logic in ingest.py."""

import math

import pytest

from congress_backtest import ingest


@pytest.mark.parametrize("band, floor, ceiling", [
    ("$1,001 - $15,000", 1001, 15000),
    ("$15,001 - $50,000", 15001, 50000),
    ("$5,000,001 - $25,000,000", 5000001, 25000000),
    ("Over $50,000,000", 50000000, None),
])
def test_parse_amount_band(band, floor, ceiling):
    assert ingest.parse_amount_band(band) == (floor, ceiling)


def test_amount_estimate_is_geometric_mean():
    # D1: geometric mean sqrt(floor * ceiling), not the arithmetic mean.
    est = ingest.amount_estimate(1001, 15000)
    assert est == pytest.approx(math.sqrt(1001 * 15000))
    # geometric mean sits below the arithmetic midpoint for a wide band
    assert est < (1001 + 15000) / 2


def test_open_top_band_scaled_from_prior_band_ratio():
    # Open band floor scaled by sqrt of the band-below's floor->ceiling ratio.
    est = ingest.amount_estimate(50_000_000, None)
    expected = 50_000_000 * math.sqrt(ingest.PRIOR_BAND_CEILING / ingest.PRIOR_BAND_FLOOR)
    assert est == pytest.approx(expected)
    assert est > 50_000_000  # always above the floor, never below it


def test_band_estimate_is_monotonic_in_floor():
    bands = ["$1,001 - $15,000", "$15,001 - $50,000",
             "$50,001 - $100,000", "$100,001 - $250,000"]
    estimates = [ingest.amount_estimate(*ingest.parse_amount_band(b)) for b in bands]
    assert estimates == sorted(estimates)
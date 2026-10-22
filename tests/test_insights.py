import pytest

from src.insights import summarize


def test_low_congestion_recommends_proceeding():
    result = summarize(occupied=20, empty=80)
    assert result.total == 100
    assert result.available == 80
    assert result.occupancy_pct == 20.0
    assert result.level == "Low"
    assert result.recommendation == "Slots available — proceed."


def test_moderate_band_includes_both_boundaries():
    at_forty = summarize(occupied=40, empty=60)
    at_seventy_five = summarize(occupied=75, empty=25)
    assert at_forty.level == "Moderate"
    assert at_forty.occupancy_pct == 40.0
    assert at_seventy_five.level == "Moderate"
    assert at_seventy_five.occupancy_pct == 75.0
    assert "nearest empty bay" in at_forty.recommendation


def test_high_congestion_sends_the_driver_elsewhere():
    just_over = summarize(occupied=76, empty=24)
    full = summarize(occupied=100, empty=0)
    assert just_over.level == "High"
    assert just_over.occupancy_pct == 76.0
    assert full.level == "High"
    assert full.available == 0
    assert full.recommendation == "Parking full — try another area."


def test_just_under_forty_is_still_low():
    result = summarize(occupied=39, empty=61)
    assert result.level == "Low"
    assert result.occupancy_pct == 39.0


def test_no_detections_is_not_treated_as_an_empty_lot():
    result = summarize(occupied=0, empty=0)
    assert result.total == 0
    assert result.level == "No slots detected"
    assert "No parking slots" in result.recommendation


def test_negative_counts_are_rejected():
    with pytest.raises(ValueError):
        summarize(occupied=-1, empty=3)

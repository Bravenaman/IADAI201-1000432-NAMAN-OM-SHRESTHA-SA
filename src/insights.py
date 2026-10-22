"""Turn slot counts into occupancy, congestion, and a driver recommendation."""

from __future__ import annotations

from dataclasses import dataclass

# Brief thresholds: low < 40%, moderate 40–75%, high > 75%.
LOW_BELOW = 40.0
HIGH_ABOVE = 75.0


@dataclass(frozen=True)
class ParkingInsights:
    total: int
    occupied: int
    available: int
    occupancy_pct: float
    level: str
    recommendation: str


def summarize(occupied: int, empty: int) -> ParkingInsights:
    """Count slots and classify congestion.

    ``occupied`` and ``empty`` are the number of detected bays of each kind.
    A lot with no detections is reported separately so the dashboard does not
    pretend the car park is empty.
    """
    if occupied < 0 or empty < 0:
        raise ValueError("Slot counts cannot be negative.")

    total = occupied + empty
    if total == 0:
        return ParkingInsights(
            total=0,
            occupied=0,
            available=0,
            occupancy_pct=0.0,
            level="No slots detected",
            recommendation=(
                "No parking slots were detected in this frame. "
                "Try a clearer camera view or a lower confidence threshold."
            ),
        )

    occupancy = occupied / total * 100.0
    if occupancy < LOW_BELOW:
        level = "Low"
        recommendation = "Slots available — proceed."
    elif occupancy <= HIGH_ABOVE:
        level = "Moderate"
        recommendation = (
            "Some slots are open — proceed, and take the nearest empty bay."
        )
    else:
        level = "High"
        recommendation = "Parking full — try another area."

    return ParkingInsights(
        total=total,
        occupied=occupied,
        available=empty,
        occupancy_pct=round(occupancy, 1),
        level=level,
        recommendation=recommendation,
    )

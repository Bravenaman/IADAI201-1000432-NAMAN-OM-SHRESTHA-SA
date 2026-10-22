import numpy as np

from src.detect import EMPTY_RGB, OCCUPIED_RGB, Slot, annotate, count_slots, slot_status


def test_pklot_class_names():
    assert slot_status("empty") == "empty"
    assert slot_status("occupied") == "occupied"
    assert slot_status("space-empty") == "empty"
    assert slot_status("space-occupied") == "occupied"


def test_unknown_class_is_rejected():
    try:
        slot_status("person")
    except ValueError as exc:
        assert "person" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_counts():
    slots = [
        Slot("empty", 0.9, (0, 0, 1, 1)),
        Slot("occupied", 0.8, (0, 0, 1, 1)),
        Slot("occupied", 0.7, (0, 0, 1, 1)),
    ]
    assert count_slots(slots) == (2, 1)


def test_overlay_uses_green_for_empty_and_red_for_occupied():
    image = np.zeros((160, 220, 3), dtype=np.uint8)
    slots = [
        Slot("empty", 0.91, (10, 20, 90, 100)),
        Slot("occupied", 0.88, (120, 20, 200, 100)),
    ]
    painted = annotate(image, slots)

    empty_edge = painted[20:24, 10:90]
    occupied_edge = painted[20:24, 120:200]
    assert empty_edge[:, :, 1].max() >= EMPTY_RGB[1] - 5
    assert occupied_edge[:, :, 0].max() >= OCCUPIED_RGB[0] - 5
    assert painted.shape == image.shape

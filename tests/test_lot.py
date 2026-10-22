import cv2
import numpy as np

from src.insights import summarize
from src.lot import load_reference, read_lot, slot_boxes


def test_grid_covers_seven_lined_rows():
    boxes = slot_boxes(640, 640)
    assert len(boxes) == 7 * 16
    assert boxes[0][1] < boxes[-1][1]


def test_empty_plate_matches_itself():
    from pathlib import Path

    plate = load_reference(Path("/workspace/samples"))
    reading = read_lot(plate, plate)
    occupied = sum(1 for slot in reading.slots if slot.status == "occupied")
    assert occupied == 0
    assert reading.same_camera


def test_cars_left_on_the_empty_frame_are_still_marked():
    from pathlib import Path

    root = Path("/workspace/samples")
    plate = load_reference(root)
    raw = cv2.imread(str(root / "lot-11.jpg"))
    occupied = sum(1 for slot in read_lot(raw, plate).slots if slot.status == "occupied")
    assert 1 <= occupied <= 6


def test_full_frame_is_busier_than_a_sparse_frame():
    from pathlib import Path

    root = Path("/workspace/samples")
    plate = load_reference(root)
    full = cv2.imread(str(root / "lot-01.jpg"))
    sparse = cv2.imread(str(root / "lot-12.jpg"))
    full_occ = sum(1 for s in read_lot(full, plate).slots if s.status == "occupied")
    sparse_occ = sum(1 for s in read_lot(sparse, plate).slots if s.status == "occupied")
    assert full_occ > 80
    assert sparse_occ < 40
    assert full_occ > sparse_occ * 3
    level = summarize(full_occ, len(slot_boxes(640, 640)) - full_occ).level
    assert level == "High"


def test_cropped_upload_is_not_reported_as_full():
    from pathlib import Path

    root = Path("/workspace/samples")
    plate = load_reference(root)
    sparse = cv2.imread(str(root / "lot-12.jpg"))
    crop = sparse[40:-50, 50:-30]
    reading = read_lot(crop, plate)
    assert reading.same_camera
    occupied = sum(1 for slot in reading.slots if slot.status == "occupied")
    assert 5 < occupied < 50
    assert reading.aligned_bgr.shape[:2] == plate.shape[:2]


def test_high_resolution_overhead_of_this_lot_is_scored():
    from pathlib import Path

    root = Path("/workspace/samples")
    plate = load_reference(root)
    sparse = cv2.imread(str(root / "lot-12.jpg"))
    high_res = cv2.resize(sparse, (sparse.shape[1] * 5, sparse.shape[0] * 5))
    reading = read_lot(high_res, plate)
    assert reading.same_camera
    occupied = sum(1 for slot in reading.slots if slot.status == "occupied")
    assert 5 < occupied < 50
    assert reading.aligned_bgr.shape[:2] == plate.shape[:2]


def test_overhead_crop_without_the_bottom_grass_is_scored():
    from pathlib import Path

    root = Path("/workspace/samples")
    plate = load_reference(root)
    sparse = cv2.imread(str(root / "lot-12.jpg"))
    height, width = sparse.shape[:2]
    crop = sparse[: int(height * 0.55), : int(width * 0.70)]
    reading = read_lot(crop, plate)
    assert reading.same_camera
    occupied = sum(1 for slot in reading.slots if slot.status == "occupied")
    empty = sum(1 for slot in reading.slots if slot.status == "empty")
    assert empty > occupied
    assert occupied < 40


def test_mirrored_lot_is_not_this_camera():
    from pathlib import Path

    plate = load_reference(Path("/workspace/samples"))
    sparse = cv2.imread("/workspace/samples/lot-12.jpg")
    reading = read_lot(cv2.flip(sparse, 1), plate)
    assert reading.same_camera is False
    assert reading.slots == []


def test_unrelated_frame_is_not_this_camera():
    from pathlib import Path

    plate = load_reference(Path("/workspace/samples"))
    noise = np.random.default_rng(0).integers(0, 255, plate.shape, dtype=np.uint8)
    reading = read_lot(noise, plate)
    assert reading.same_camera is False
    assert reading.slots == []

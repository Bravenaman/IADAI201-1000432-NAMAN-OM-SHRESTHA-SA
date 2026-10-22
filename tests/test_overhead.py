import cv2
import numpy as np

from src.overhead import read_overhead


def test_flipped_sparse_lot_is_mostly_empty():
    image = cv2.imread("/workspace/samples/lot-12.jpg")
    reading = read_overhead(cv2.flip(image, 1))
    assert reading.same_camera is False
    occupied = sum(1 for slot in reading.slots if slot.status == "occupied")
    empty = sum(1 for slot in reading.slots if slot.status == "empty")
    assert empty > 40
    assert occupied < 40
    assert empty > occupied * 3


def test_sideways_sparse_lot_is_still_read():
    image = cv2.imread("/workspace/samples/lot-12.jpg")
    sideways = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    reading = read_overhead(sideways)
    occupied = sum(1 for slot in reading.slots if slot.status == "occupied")
    empty = sum(1 for slot in reading.slots if slot.status == "empty")
    assert empty > occupied
    assert empty > 40


def test_full_overhead_lot_is_mostly_occupied():
    image = cv2.imread("/workspace/samples/lot-01.jpg")
    reading = read_overhead(cv2.flip(image, 1))
    occupied = sum(1 for slot in reading.slots if slot.status == "occupied")
    empty = sum(1 for slot in reading.slots if slot.status == "empty")
    assert occupied > 60
    assert occupied > empty * 3


def test_topdown_lined_lot_marks_empty_and_occupied_bays():
    image = cv2.imread("/workspace/tests/fixtures/topdown-lot.jpg")
    reading = read_overhead(image)
    assert reading.source == "lines"
    occupied = sum(1 for slot in reading.slots if slot.status == "occupied")
    empty = sum(1 for slot in reading.slots if slot.status == "empty")
    assert empty > 25
    assert occupied > 30
    assert empty + occupied > 80


def test_sideways_topdown_lot_is_still_read():
    from src.lined import read_lined

    image = cv2.imread("/workspace/tests/fixtures/topdown-lot.jpg")
    sideways = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    slots, _ = read_lined(sideways)
    occupied = sum(1 for slot in slots if slot.status == "occupied")
    empty = sum(1 for slot in slots if slot.status == "empty")
    assert empty > 25
    assert occupied > 30


def test_noise_has_no_bays():
    image = cv2.imread("/workspace/samples/lot-12.jpg")
    noise = np.random.default_rng(1).integers(0, 255, image.shape, dtype=np.uint8)
    reading = read_overhead(noise)
    assert reading.slots == []

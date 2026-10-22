"""Find empty and occupied bays in an overhead shot of any lined lot.

The sample camera has a saved bay map. A different overhead photo does not
use that map. A PKLot parking-space model marks each visible bay instead,
so a shot from another angle is still scored.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from src.detect import Slot
from src.lined import read_lined
from src.lot import LotReading

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pklot-yolov8n.pt"
# Trained at 960. Lower confidence keeps faint bays; the model is already strict.
_CONFIDENCE = 0.30
_IMAGE_SIZE = 960
_MIN_BAYS = 4


@lru_cache(maxsize=1)
def load_model():
    """Load the parking-space weights once per process."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Parking model not found at {MODEL_PATH}")
    from ultralytics import YOLO

    return YOLO(str(MODEL_PATH))


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def _dedupe(slots: list[Slot]) -> list[Slot]:
    """Drop a weaker box that lands on the same bay."""
    ordered = sorted(slots, key=lambda slot: slot.confidence, reverse=True)
    kept: list[Slot] = []
    for slot in ordered:
        if any(_iou(slot.xyxy, other.xyxy) > 0.55 for other in kept):
            continue
        kept.append(slot)
    return kept


def _detect(image_bgr: np.ndarray) -> list[Slot]:
    result = load_model().predict(
        image_bgr,
        verbose=False,
        conf=_CONFIDENCE,
        iou=0.45,
        imgsz=_IMAGE_SIZE,
    )[0]
    names = result.names
    height, width = image_bgr.shape[:2]
    frame = float(height * width)
    slots: list[Slot] = []
    if result.boxes is None:
        return slots
    for box in result.boxes:
        label = str(names[int(box.cls[0])]).lower()
        status = "empty" if "vacant" in label or "empty" in label else "occupied"
        x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        area = max(0, x2 - x1) * max(0, y2 - y1)
        # A bay is a small patch of the frame. A box over the whole road is not one.
        if area < frame * 0.00015 or area > frame * 0.03:
            continue
        if x2 - x1 < 6 or y2 - y1 < 6:
            continue
        slots.append(Slot(status=status, confidence=float(box.conf[0]), xyxy=(x1, y1, x2, y2)))
    return _dedupe(slots)


def _orientations(image_bgr: np.ndarray):
    """Upright first, then turns. A sideways overhead shot is tried on its side."""
    yield image_bgr
    yield cv2.rotate(image_bgr, cv2.ROTATE_180)
    yield cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
    yield cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)


def read_overhead(image_bgr: np.ndarray) -> LotReading:
    """Mark bays in an overhead photo that is not the sample camera.

    A surveillance-style lot uses the PKLot model. A top-down lot, where
    that model sees nothing, is read from the painted stall dividers.
    """
    best_slots: list[Slot] = []
    best_image = image_bgr
    for view in _orientations(image_bgr):
        slots = _detect(view)
        if len(slots) > len(best_slots):
            best_slots = slots
            best_image = view
        if len(slots) >= 15:
            best_slots = slots
            best_image = view
            break
    if len(best_slots) >= _MIN_BAYS:
        return LotReading(slots=best_slots, same_camera=False, aligned_bgr=best_image, source="model")

    lined_slots, lined_image = read_lined(image_bgr)
    if len(lined_slots) >= _MIN_BAYS:
        return LotReading(slots=lined_slots, same_camera=False, aligned_bgr=lined_image, source="lines")
    return LotReading(slots=[], same_camera=False, aligned_bgr=image_bgr, source="lines")

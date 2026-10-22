"""Read a top-down lot from the painted stall dividers.

The PKLot model knows the surveillance-camera angle. A straight overhead
photo, with the bays drawn on the asphalt, is a different picture. The
dividers are traced, each gap of one stall is a bay, and a bay that holds
something other than bare asphalt is occupied.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.detect import Slot

_MIN_BAYS = 8


def _segments(gray: np.ndarray) -> list[tuple[float, float, float, float, float]]:
    raw = cv2.createLineSegmentDetector(0).detect(gray)[0]
    if raw is None:
        return []
    found = []
    for x1, y1, x2, y2 in raw.reshape(-1, 4):
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < 16:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if angle > 90:
            angle = 180 - angle
        found.append((float(x1), float(y1), float(x2), float(y2), angle, length))
    return found


def _divider_weight(gray: np.ndarray, vertical: bool) -> float:
    total = 0.0
    for *_, angle, length in _segments(gray):
        steep = angle >= 55
        if steep == vertical and 16 <= length <= 120:
            total += length
    return total


def _dividers(gray: np.ndarray) -> list[dict]:
    divs = []
    for x1, y1, x2, y2, angle, length in _segments(gray):
        if length > 110 or angle < 55:
            continue
        divs.append(
            {
                "x": (x1 + x2) / 2,
                "y": (y1 + y2) / 2,
                "y1": min(y1, y2),
                "y2": max(y1, y2),
            }
        )
    return divs


def _row_centers(divs: list[dict], height: int) -> list[float]:
    if len(divs) < 8:
        return []
    ys = np.array([d["y"] for d in divs])
    bins = np.arange(0, height + 8, 8)
    hist, edges = np.histogram(ys, bins=bins)
    smooth = np.convolve(hist.astype(float), np.ones(3) / 3, mode="same")
    peaks = []
    for index in range(1, len(smooth) - 1):
        if smooth[index] >= smooth[index - 1] and smooth[index] >= smooth[index + 1] and hist[index] >= 5:
            peaks.append(float((edges[index] + edges[index + 1]) / 2))
    merged: list[float] = []
    for peak in peaks:
        if merged and peak - merged[-1] < 70:
            merged[-1] = (merged[-1] + peak) / 2
        else:
            merged.append(peak)
    return merged


def _boxes_for_row(divs: list[dict], center: float, height: int, green: np.ndarray) -> list[tuple[int, int, int, int]]:
    items = [d for d in divs if abs(d["y"] - center) < 48]
    if len(items) < 6:
        return []
    items.sort(key=lambda d: d["x"])
    cols: list[dict] = []
    for item in items:
        if cols and item["x"] - cols[-1]["x"] < 8:
            col = cols[-1]
            col["x"] = (col["x"] * col["n"] + item["x"]) / (col["n"] + 1)
            col["n"] += 1
        else:
            cols.append({"x": item["x"], "n": 1})
    gaps = np.diff([col["x"] for col in cols])
    plausible = gaps[(gaps > 12) & (gaps < 70)]
    if len(plausible) < 3:
        return []
    pitch = float(np.median(plausible))
    half = int(pitch * 1.7)
    y1, y2 = int(center - half), int(center + half)
    boxes = []
    for index in range(len(cols) - 1):
        gap = cols[index + 1]["x"] - cols[index]["x"]
        count = int(round(gap / pitch))
        if not 1 <= count <= 2:
            continue
        step = gap / count
        if step < pitch * 0.6 or step > pitch * 1.45:
            continue
        for slot_index in range(count):
            left = cols[index]["x"] + step * slot_index
            right = left + step
            box = (
                int(round(left)) + 2,
                max(0, y1 + 2),
                int(round(right)) - 2,
                min(height - 1, y2 - 2),
            )
            if box[2] - box[0] < 8 or box[3] - box[1] < 14:
                continue
            patch = green[box[1] : box[3], box[0] : box[2]]
            if patch.size and float(patch.mean()) > 0.22:
                continue
            boxes.append(box)
    return boxes


def _vehicle_fraction(image: np.ndarray, box: tuple[int, int, int, int], asphalt: float) -> float:
    x1, y1, x2, y2 = box
    dx = max(1, (x2 - x1) // 10)
    dy = max(1, (y2 - y1) // 8)
    crop = image[y1 + dy : y2 - dy, x1 + dx : x2 - dx]
    if crop.size < 16:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # A stall divider is a thin bright stroke. A car is a wide patch.
    paint = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    differ = np.abs(gray.astype(np.int16) - asphalt) > 32
    colored = hsv[:, :, 1] > 40
    vehicle = (differ | colored) & (paint <= 18)
    vehicle = cv2.morphologyEx(vehicle.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return float(vehicle.mean())


def _asphalt_level(image: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> float:
    spreads = []
    for box in boxes:
        gray = cv2.cvtColor(image[box[1] : box[3], box[0] : box[2]], cv2.COLOR_BGR2GRAY)
        spreads.append(float(gray.std()))
    spreads = np.array(spreads)
    calm = spreads <= np.percentile(spreads, 30)
    levels = []
    for box, quiet in zip(boxes, calm):
        if not quiet:
            continue
        gray = cv2.cvtColor(image[box[1] : box[3], box[0] : box[2]], cv2.COLOR_BGR2GRAY)
        levels.append(float(np.median(gray)))
    if not levels:
        return 140.0
    return float(np.median(levels))


def _keep_runs(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Drop a pair of boxes sitting on the road, away from a real row."""
    if not boxes:
        return []
    widths = [box[2] - box[0] for box in boxes]
    gap_limit = float(np.median(widths)) * 2.4
    rows: list[list[tuple[int, int, int, int]]] = []
    for box in sorted(boxes, key=lambda item: (item[1] + item[3]) / 2):
        center = (box[1] + box[3]) / 2
        if rows and abs(center - np.mean([(item[1] + item[3]) / 2 for item in rows[-1]])) < 24:
            rows[-1].append(box)
        else:
            rows.append([box])
    kept = []
    for row in rows:
        row.sort(key=lambda item: item[0])
        run = [row[0]]
        runs = []
        for box in row[1:]:
            if box[0] - run[-1][2] > gap_limit:
                runs.append(run)
                run = [box]
            else:
                run.append(box)
        runs.append(run)
        for run in runs:
            if len(run) >= 3:
                kept.extend(run)
    return kept


def _slots_upright(image: np.ndarray) -> list[Slot]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = (hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 95) & (hsv[:, :, 1] > 40)
    divs = _dividers(gray)
    boxes: list[tuple[int, int, int, int]] = []
    for center in _row_centers(divs, image.shape[0]):
        boxes.extend(_boxes_for_row(divs, center, image.shape[0], green))
    boxes = _keep_runs(boxes)
    if len(boxes) < _MIN_BAYS:
        return []
    asphalt = _asphalt_level(image, boxes)
    slots = []
    for box in boxes:
        fraction = _vehicle_fraction(image, box, asphalt)
        occupied = fraction > 0.18
        slots.append(
            Slot(
                status="occupied" if occupied else "empty",
                confidence=float(np.clip(fraction if occupied else 1.0 - fraction, 0.0, 0.99)),
                xyxy=box,
            )
        )
    return slots


def _unmap_clockwise(box: tuple[int, int, int, int], height: int) -> tuple[int, int, int, int]:
    """Map a box from a 90° clockwise view back onto the original photo."""
    x1, y1, x2, y2 = box
    corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    original = [(y, height - 1 - x) for x, y in corners]
    xs = [point[0] for point in original]
    ys = [point[1] for point in original]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def read_lined(image_bgr: np.ndarray) -> tuple[list[Slot], np.ndarray]:
    """Return bays traced from painted dividers, and the image they sit on."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    vertical = _divider_weight(gray, vertical=True)
    horizontal = _divider_weight(gray, vertical=False)
    if horizontal > vertical * 1.25 and horizontal > 200:
        turned = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
        slots = _slots_upright(turned)
        if len(slots) >= _MIN_BAYS:
            height = image_bgr.shape[0]
            mapped = [
                Slot(slot.status, slot.confidence, _unmap_clockwise(slot.xyxy, height))
                for slot in slots
            ]
            return mapped, image_bgr
    return _slots_upright(image_bgr), image_bgr

"""ParkVision AI.

One Streamlit file. Upload an overhead shot of a lined parking lot, or open a
sample frame. Bays that match the sample camera use that camera's map. Any
other overhead shot is read from a parking-space model, or from the painted
stall lines when the shot is straight down. On a diagonal or herringbone lot,
each stall is boxed along its row. An open bay between two cars is marked
empty, and so is an open stall whose painted sides are still visible.
"""

from __future__ import annotations

import base64
import io
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"


# RGB colours required by the brief: green empty, red occupied.
EMPTY_RGB = (34, 197, 94)
OCCUPIED_RGB = (239, 68, 68)

_EMPTY_HINTS = ("empty", "free", "vacant", "available")
_OCCUPIED_HINTS = ("occupied", "busy", "taken", "full", "car")


@dataclass(frozen=True)
class Slot:
    status: str
    confidence: float
    xyxy: tuple[int, int, int, int]
    # Four corners when the bay is slanted. Straight bays leave this empty
    # and are drawn from xyxy.
    quad: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None = None


def slot_status(class_name: str) -> str:
    """Map a detector class name to ``empty`` or ``occupied``.

    PKLot exports use names such as ``empty`` / ``occupied`` or
    ``space-empty`` / ``space-occupied``. Empty is checked first so a name
    that somehow contains both tokens is not marked occupied.
    """
    name = class_name.lower().replace("_", " ").replace("-", " ")
    if any(hint in name for hint in _EMPTY_HINTS):
        return "empty"
    if any(hint in name for hint in _OCCUPIED_HINTS):
        return "occupied"
    raise ValueError(f"Unrecognised parking class: {class_name!r}")


def slots_from_boxes(names: dict, boxes) -> list[Slot]:
    """Read Ultralytics boxes into slot records.

    ``names`` is the model's ``result.names`` mapping. ``boxes`` is
    ``result.boxes``.
    """
    slots: list[Slot] = []
    if boxes is None or len(boxes) == 0:
        return slots
    for box in boxes:
        class_id = int(box.cls[0])
        class_name = str(names[class_id])
        x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0].tolist())
        slots.append(
            Slot(
                status=slot_status(class_name),
                confidence=float(box.conf[0]),
                xyxy=(x1, y1, x2, y2),
            )
        )
    return slots


def annotate(image_rgb: np.ndarray, slots: list[Slot]) -> np.ndarray:
    """Return a copy of ``image_rgb`` with colour-coded slot boxes."""
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    height = canvas.shape[0]
    thickness = max(2, height // 320)
    font_scale = max(0.4, height / 1400)

    for slot in slots:
        rgb = EMPTY_RGB if slot.status == "empty" else OCCUPIED_RGB
        color = (rgb[2], rgb[1], rgb[0])
        x1, y1, x2, y2 = slot.xyxy
        if slot.quad is not None:
            corners = np.array(slot.quad, dtype=np.int32)
            cv2.polylines(canvas, [corners], True, color, thickness)
            x1 = int(min(point[0] for point in slot.quad))
            y1 = int(min(point[1] for point in slot.quad))
        else:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        label = "occ" if slot.status == "occupied" else "empty"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        if (x2 - x1) >= tw + 8:
            top = max(0, y1 - th - baseline - 4)
            cv2.rectangle(
                canvas, (x1, top), (x1 + tw + 6, top + th + baseline + 4), color, -1
            )
            cv2.putText(
                canvas,
                label,
                (x1 + 3, top + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def count_slots(slots: list[Slot]) -> tuple[int, int]:
    occupied = sum(1 for slot in slots if slot.status == "occupied")
    empty = sum(1 for slot in slots if slot.status == "empty")
    return occupied, empty


# Used only when the sample frames cannot be read. The live limit is the
# mean number of open bays across those frames.
_FALLBACK_OPEN_FRACTION = 46 / 144
_SAMPLE_RATE = 22050


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
    share_open = empty / total
    average_open = open_bay_average().fraction
    # Free sits at twice the usual number of open bays, full at half of it.
    if share_open >= average_open * 2:
        level = "Low"
        recommendation = "Slots available — proceed."
    elif share_open <= average_open * 0.5:
        level = "High"
        recommendation = "Parking full — try another area."
    else:
        level = "Moderate"
        recommendation = (
            "Some slots are open — proceed, and take the nearest empty bay."
        )

    return ParkingInsights(
        total=total,
        occupied=occupied,
        available=empty,
        occupancy_pct=round(occupancy, 1),
        level=level,
        recommendation=recommendation,
    )


# Row bands (top, bottom) as fractions of the frame. They sit on the lined
# asphalt rows and skip the driving aisles.
ROW_BANDS: tuple[tuple[float, float], ...] = (
    (0.000, 0.070),
    (0.075, 0.150),
    (0.160, 0.245),
    (0.255, 0.340),
    (0.360, 0.430),
    (0.455, 0.540),
    (0.555, 0.650),
)
# The cobblestone row under the grass median, then the row at the bottom
# edge of the frame. The driving aisle between them is not a bay.
LOWER_BAND = (0.692, 0.782)
BOTTOM_BAND = (0.860, 0.970)
# Stop before the drive lane on the right. That asphalt matches an empty
# bay and was being reported as free space. The cobblestone rows end at the kerb.
X_START = 0.00
X_STOP = 0.80
LOWER_X_STOP = 0.78
X_STEP = 0.05

# Below this normalised correlation the bay is occupied.
OCCUPIED_BELOW = 0.55
# The lower row is mottled cobblestone, so a sunny empty bay matches the
# plate a little less well than the asphalt rows do.
LOWER_OCCUPIED_BELOW = 0.50
# Tree shadows on the bottom row fail to match the plate without a car
# being there. A mild mismatch counts only when the bay is also much busier
# than bare cobblestone. A very poor match is a vehicle on its own.
BOTTOM_MATCH = 0.28
BOTTOM_STD_MATCH = 0.55
BOTTOM_STD = 38.0
# A bay that still resembles the plate, but is much busier, is a partial vehicle.
TEXTURE_RATIO = 4.5
TEXTURE_MARGIN = 0.23

REFERENCE_NAME = "lot-11.jpg"
_ROW_COUNT = len(ROW_BANDS) + 2


@dataclass(frozen=True)
class LotReading:
    slots: list[Slot]
    same_camera: bool
    aligned_bgr: np.ndarray
    # "camera" uses the saved sample map, "model" the PKLot detector,
    # "lines" the painted dividers in a top-down shot.
    source: str = "camera"


def _x_edges(width: int, x_stop: float) -> list[int]:
    edges: list[float] = []
    x = X_START
    while x < x_stop - 1e-6:
        edges.append(x)
        x = round(x + X_STEP, 5)
    edges.append(x_stop)
    return [int(round(value * width)) for value in edges]


def slot_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Pixel boxes for every lined bay, left to right, top to bottom."""
    boxes: list[tuple[int, int, int, int]] = []
    spans = [((top, bottom), X_STOP) for top, bottom in ROW_BANDS]
    spans.append((LOWER_BAND, LOWER_X_STOP))
    spans.append((BOTTOM_BAND, LOWER_X_STOP))
    for (y1, y2), x_stop in spans:
        top = int(round(y1 * height))
        bottom = int(round(y2 * height))
        edges = _x_edges(width, x_stop)
        for left, right in zip(edges, edges[1:]):
            if right - left >= 2 and bottom - top >= 2:
                boxes.append((left, top, right, bottom))
    return boxes


def _crop(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    dx = max(1, (x2 - x1) // 10)
    dy = max(1, (y2 - y1) // 8)
    return image[y1 + dy : y2 - dy, x1 + dx : x2 - dx]


def _correlation(image: np.ndarray, reference: np.ndarray, box: tuple[int, int, int, int]) -> float:
    current = _crop(image, box)
    baseline = _crop(reference, box)
    if current.size < 16 or current.shape != baseline.shape:
        return 1.0
    current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    baseline = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
    score = cv2.matchTemplate(current, baseline, cv2.TM_CCOEFF_NORMED)
    return float(score[0, 0])


def _laplacian(image: np.ndarray, box: tuple[int, int, int, int]) -> float:
    gray = cv2.cvtColor(_crop(image, box), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def _texture_ratio(image: np.ndarray, reference: np.ndarray, box: tuple[int, int, int, int]) -> float:
    return _laplacian(image, box) / (_laplacian(reference, box) + 1.0)


def _parked_on_plate(image: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """True when a vehicle is sitting in this bay of the empty plate.

    Border bays are skipped. The frame edge and the kerb look busy without
    being a car, and repainting them marks every later frame occupied.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    if x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1:
        return False
    if _laplacian(image, box) < 140:
        return False
    gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    dark = (gray.astype(np.int16) < np.median(gray) - 18).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    if count <= 1:
        return False
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[index, cv2.CC_STAT_AREA] / float(dark.size)
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    return (
        area > 0.18
        and stats[index, cv2.CC_STAT_WIDTH] / box_w > 0.45
        and stats[index, cv2.CC_STAT_HEIGHT] / box_h > 0.45
    )


def _clear_parked_vehicles(plate: np.ndarray) -> np.ndarray:
    """Replace vehicles parked on the empty plate with a neighboring empty bay."""
    boxes = slot_boxes(plate.shape[1], plate.shape[0])
    if not boxes:
        return plate
    per_row = len(boxes) // _ROW_COUNT
    energy = [_laplacian(plate, box) for box in boxes]
    cleaned = plate.copy()
    for index, box in enumerate(boxes):
        if not _parked_on_plate(plate, box):
            continue
        row, col = divmod(index, per_row)
        calm = float(np.median(energy[row * per_row : (row + 1) * per_row]))
        donor = None
        for other_col in range(per_row):
            other = row * per_row + other_col
            if other == index or energy[other] > calm * 1.4:
                continue
            distance = abs(other_col - col)
            if donor is None or distance < donor[0]:
                donor = (distance, other)
        if donor is None:
            continue
        x1, y1, x2, y2 = box
        sx1, sy1, sx2, sy2 = boxes[donor[1]]
        patch = plate[sy1:sy2, sx1:sx2]
        if patch.shape[0] != y2 - y1 or patch.shape[1] != x2 - x1:
            patch = cv2.resize(patch, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
        cleaned[y1:y2, x1:x2] = patch
    return cleaned


def _grass_correlation(image: np.ndarray, reference: np.ndarray) -> float:
    """The grass median between the last row and the lower court should match."""
    height, width = reference.shape[:2]
    box = (
        int(0.20 * width),
        int(0.655 * height),
        int(0.60 * width),
        int(0.705 * height),
    )
    return _correlation(image, reference, box)


def _prepare(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(3.0, (8, 8)).apply(gray)


def _align_to_reference(image: np.ndarray, reference: np.ndarray) -> np.ndarray | None:
    """Warp an overhead shot into the empty plate's pixel grid.

    Uploads are often a higher resolution than the 640px plate, a crop that
    leaves out the bottom grass, or a frame with a border. Stretching those
    onto 640×640 slides the bay grid onto the road.
    """
    # Match near the plate's size. ORB's pyramid does not span a 12MP
    # overhead shot down to a 640px plate, so the shot is reduced first.
    reference_max = max(reference.shape[:2])
    image_max = max(image.shape[:2])
    match_image = image
    shrink = 1.0
    if image_max > reference_max * 2:
        shrink = (reference_max * 2) / float(image_max)
        match_image = cv2.resize(
            image,
            (max(1, int(image.shape[1] * shrink)), max(1, int(image.shape[0] * shrink))),
            interpolation=cv2.INTER_AREA,
        )

    gray_image = _prepare(cv2.cvtColor(match_image, cv2.COLOR_BGR2GRAY))
    gray_reference = _prepare(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY))
    orb = cv2.ORB_create(8000, fastThreshold=8)
    keys_image, desc_image = orb.detectAndCompute(gray_image, None)
    keys_reference, desc_reference = orb.detectAndCompute(gray_reference, None)
    if desc_image is None or desc_reference is None or len(keys_image) < 20 or len(keys_reference) < 20:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(desc_image, desc_reference, k=2)
    good = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < 0.75 * second.distance:
            good.append(best)
    if len(good) < 12:
        return None

    source = np.float32([keys_image[match.queryIdx].pt for match in good])
    destination = np.float32([keys_reference[match.trainIdx].pt for match in good])
    homography, mask = cv2.findHomography(source, destination, cv2.RANSAC, 5.0)
    if homography is not None and shrink != 1.0:
        homography = homography @ np.array(
            [[shrink, 0.0, 0.0], [0.0, shrink, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    inliers = 0 if mask is None else int(mask.sum())
    if homography is None or inliers < 12:
        return None
    # A mirrored photo is a different view. Do not force the bay map onto it.
    if float(np.linalg.det(homography[:2, :2])) <= 0:
        return None
    scales = np.linalg.svd(homography[:2, :2], compute_uv=False)
    # A phone photo of this lot is several times the 640px plate. That still
    # lines up; only an extreme resize is rejected.
    if float(scales.min()) < 0.08 or float(scales.max()) > 12.0:
        return None

    height, width = reference.shape[:2]
    warped = cv2.warpPerspective(
        image,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    covered = np.count_nonzero(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 8) / float(height * width)
    grass = _grass_correlation(warped, reference)
    # A full frame should reproduce the grass strip. A closer overhead shot
    # often cuts that strip off, and a strong match is enough on its own.
    strong = inliers >= 28 and covered >= 0.18
    weak = covered >= 0.45 and grass >= 0.18
    if not (strong or weak):
        return None
    return warped


def _registered(image: np.ndarray, reference: np.ndarray) -> bool:
    if image.shape[:2] != reference.shape[:2]:
        return False
    return _grass_correlation(image, reference) >= 0.20


def read_lot(image_bgr: np.ndarray, reference_bgr: np.ndarray, occupied_below: float = OCCUPIED_BELOW) -> LotReading:
    """Mark every calibrated bay occupied or empty."""
    matched = _registered(image_bgr, reference_bgr)
    if not matched:
        aligned = _align_to_reference(image_bgr, reference_bgr)
        if aligned is None:
            return LotReading(slots=[], same_camera=False, aligned_bgr=image_bgr)
        image_bgr = aligned
        matched = True
    grass = _grass_correlation(image_bgr, reference_bgr)
    boxes = slot_boxes(image_bgr.shape[1], image_bgr.shape[0])
    height = image_bgr.shape[0]
    lower_cut = int(0.68 * height)
    bottom_cut = int(0.82 * height)
    slots: list[Slot] = []
    for box in boxes:
        x1, y1, x2, y2 = box
        bay = image_bgr[y1:y2, x1:x2]
        if bay.size == 0:
            continue
        gray = cv2.cvtColor(bay, cv2.COLOR_BGR2GRAY)
        covered = np.count_nonzero(gray > 8) / float(bay.shape[0] * bay.shape[1])
        if covered < 0.65:
            continue
        match = _correlation(image_bgr, reference_bgr, box)
        if y1 >= bottom_cut:
            # Branches move between frames, so busyness against the plate
            # paints an empty bottom stall red. A car is a solid mismatch.
            spread = float(gray.std())
            occupied = match < BOTTOM_MATCH or (match < BOTTOM_STD_MATCH and spread > BOTTOM_STD)
        else:
            limit = min(occupied_below, LOWER_OCCUPIED_BELOW) if y1 >= lower_cut else occupied_below
            soft_below = min(0.85, limit + TEXTURE_MARGIN)
            busy = _texture_ratio(image_bgr, reference_bgr, box)
            occupied = match < limit or (match < soft_below and busy > TEXTURE_RATIO)
        confidence = (1.0 - match) if occupied else match
        confidence = float(np.clip(confidence, 0.0, 0.99))
        slots.append(
            Slot(
                status="occupied" if occupied else "empty",
                confidence=confidence,
                xyxy=box,
            )
        )
    return LotReading(slots=slots, same_camera=matched or grass >= 0.20, aligned_bgr=image_bgr)


def load_reference(samples_dir: Path) -> np.ndarray:
    """Empty plate for this camera, with any parked vehicles painted out."""
    path = samples_dir / REFERENCE_NAME
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Empty-lot plate not found at {path}")
    return _clear_parked_vehicles(image)


_LINED_MIN_BAYS = 8


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
    # A stall divider is a thin bright stroke. The kernel stays narrower than
    # the bay, or a whole car in a tight stall is mistaken for the line.
    span = max(3, min(9, (x2 - x1) // 3))
    if span % 2 == 0:
        span += 1
    paint = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (span, span)))
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
    if len(boxes) < _LINED_MIN_BAYS:
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


def _rotate_bound(image: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate without clipping, and return the affine map back to the original."""
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    bound_w = int(height * sine + width * cosine)
    bound_h = int(height * cosine + width * sine)
    matrix[0, 2] += bound_w / 2 - center[0]
    matrix[1, 2] += bound_h / 2 - center[1]
    warped = cv2.warpAffine(image, matrix, (bound_w, bound_h), flags=cv2.INTER_LINEAR)
    return warped, matrix


def _line_angle(dx: float, dy: float) -> float:
    """Undirected angle in degrees, from 0 up to 180."""
    return float(np.degrees(np.arctan2(dy, dx)) % 180.0)


def _angle_delta(a: float, b: float) -> float:
    delta = abs(a - b) % 180.0
    return min(delta, 180.0 - delta)


def _ridge_strength(gray: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> float:
    """How much brighter the stroke is than the asphalt on either side.

    A stall line is a thin bright ridge. The edge of a roof is a step, and
    that scores low here, so it is not treated as a divider.
    """
    height, width = gray.shape
    ang = np.arctan2(y2 - y1, x2 - x1)
    nx, ny = -np.sin(ang), np.cos(ang)
    scores = []
    for t in (0.25, 0.5, 0.75):
        mx = x1 + (x2 - x1) * t
        my = y1 + (y2 - y1) * t
        vals = []
        for step in range(-6, 7):
            x = int(round(mx + nx * step))
            y = int(round(my + ny * step))
            if not (0 <= x < width and 0 <= y < height):
                vals = []
                break
            vals.append(int(gray[y, x]))
        if len(vals) < 13:
            continue
        center = float(np.mean(vals[5:8]))
        sides = 0.5 * (float(np.mean(vals[0:3])) + float(np.mean(vals[10:13])))
        scores.append(center - sides)
    if len(scores) < 2:
        return -99.0
    return float(np.median(scores))


def _paint_segments(gray: np.ndarray):
    kept = []
    for x1, y1, x2, y2, _folded, length in _segments(gray):
        if length < 22 or _ridge_strength(gray, x1, y1, x2, y2) < 6:
            continue
        kept.append((x1, y1, x2, y2, _line_angle(x2 - x1, y2 - y1), length))
    return kept


def _slant_peaks(gray: np.ndarray) -> list[float]:
    """Directions of diagonal stall dividers, ignoring upright and flat paint."""
    hist = np.zeros(180, dtype=np.float64)
    for *_, ang, length in _paint_segments(gray):
        hist[int(ang) % 180] += length
    smooth = np.convolve(hist, np.ones(9) / 9, mode="same")
    for degree in range(180):
        if _angle_delta(degree, 0) < 18 or _angle_delta(degree, 90) < 18:
            smooth[degree] = 0
    found = []
    for degree in range(180):
        if smooth[degree] < 220:
            continue
        if smooth[degree] >= smooth[(degree - 1) % 180] and smooth[degree] > smooth[(degree + 1) % 180]:
            found.append((float(smooth[degree]), float(degree)))
    found.sort(reverse=True)
    kept: list[float] = []
    for _weight, degree in found:
        if all(_angle_delta(degree, other) > 18 for other in kept):
            kept.append(degree)
    # A line within 25° of the frame is a straight lot seen on its side.
    # Treating that as a slant boxes a handful of cars and skips the real grid.
    return [degree for degree in kept[:3] if _angle_delta(degree, 0) >= 25 and _angle_delta(degree, 90) >= 25]


def _dividers_at(gray: np.ndarray, degree: float) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Painted strokes along one slant, as a position across the row and a span along it."""
    direction = np.array([np.cos(np.radians(degree)), np.sin(np.radians(degree))])
    normal = np.array([-direction[1], direction[0]])
    raw = []
    for x1, y1, x2, y2, ang, length in _paint_segments(gray):
        if length < 18 or _angle_delta(ang, degree) > 11:
            continue
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        span = sorted((float(np.dot(p1, direction)), float(np.dot(p2, direction))))
        raw.append(
            {
                "t0": span[0],
                "t1": span[1],
                "perp": float(np.dot((p1 + p2) / 2, normal)),
            }
        )
    merged: list[dict] = []
    for stroke in raw:
        host = next(
            (
                item
                for item in merged
                if abs(stroke["perp"] - item["perp"]) <= 6
                and stroke["t0"] <= item["t1"] + 14
                and item["t0"] <= stroke["t1"] + 14
            ),
            None,
        )
        if host is None:
            merged.append(dict(stroke))
            continue
        host["t0"] = min(host["t0"], stroke["t0"])
        host["t1"] = max(host["t1"], stroke["t1"])
        host["perp"] = (host["perp"] + stroke["perp"]) / 2
    return merged, direction, normal


def _stall_pitch(lines: list[dict]) -> float | None:
    """Width of one bay: the gap to the next divider, skipping a line's own edge."""
    gaps = []
    for left in lines:
        best = None
        for right in lines:
            dist = abs(right["perp"] - left["perp"])
            if dist < 30:
                continue
            overlap = min(left["t1"], right["t1"]) - max(left["t0"], right["t0"])
            shorter = min(left["t1"] - left["t0"], right["t1"] - right["t0"])
            if shorter < 16 or overlap < 0.35 * shorter:
                continue
            if best is None or dist < best:
                best = dist
        if best is not None and best <= 75:
            gaps.append(best)
    if len(gaps) < 3:
        return None
    return float(np.median(gaps))


def _bay_span(left: dict, partner: dict, pitch: float) -> tuple[float, float] | None:
    """Place the bay on the stall, not on the aisle in front of it.

    The two dividers overlap on the paint that is actually visible. That
    overlap is often only the entrance tick, which sits in the driving
    lane. The rest of the stall is the side where those lines continue.
    """
    overlap0 = max(left["t0"], partner["t0"])
    overlap1 = min(left["t1"], partner["t1"])
    if overlap1 - overlap0 < pitch * 0.45:
        return None
    union0 = min(left["t0"], partner["t0"])
    union1 = max(left["t1"], partner["t1"])
    target = pitch * 2.25
    if union1 - union0 <= target:
        if union1 - union0 < pitch * 1.45:
            return None
        return union0, union1
    extra_lo = overlap0 - union0
    extra_hi = union1 - overlap1
    if extra_lo >= extra_hi:
        t1 = overlap1
        t0 = max(union0, t1 - target)
    else:
        t0 = overlap0
        t1 = min(union1, t0 + target)
    if t1 - t0 < pitch * 1.45:
        return None
    return t0, t1


def _quads_for_family(image: np.ndarray, degree: float) -> list[tuple]:
    """One bay between each pair of neighboring dividers on this slant.

    The bay is the painted gap between two stall lines. It follows those
    lines back onto the stall, and it is not left on the entrance tick in
    the driving aisle.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lines, direction, normal = _dividers_at(gray, degree)
    pitch = _stall_pitch(lines)
    if pitch is None:
        return []
    pieces = []
    for line in lines:
        length = line["t1"] - line["t0"]
        if not pitch * 0.55 <= length <= pitch * 3.3:
            continue
        pieces.append(line)
    if len(pieces) < 4:
        return []
    height, width = image.shape[:2]
    quads = []
    used: set[tuple] = set()
    for left in pieces:
        partner = None
        partner_dist = 1e9
        for right in pieces:
            dist = right["perp"] - left["perp"]
            if not pitch * 0.72 <= dist <= pitch * 1.28 or dist >= partner_dist:
                continue
            blocked = False
            for other in pieces:
                if not left["perp"] + dist * 0.34 < other["perp"] < right["perp"] - dist * 0.34:
                    continue
                shared = min(left["t1"], right["t1"], other["t1"]) - max(left["t0"], right["t0"], other["t0"])
                if shared > pitch * 0.25:
                    blocked = True
                    break
            if blocked:
                continue
            overlap = min(left["t1"], right["t1"]) - max(left["t0"], right["t0"])
            if overlap < pitch * 0.45:
                continue
            partner, partner_dist = right, dist
        if partner is None:
            continue
        span = _bay_span(left, partner, pitch)
        if span is None:
            continue
        t0, t1 = span
        key = (round(left["perp"], 1), round(partner["perp"], 1), round((t0 + t1) / 2))
        if key in used:
            continue
        used.add(key)
        corners = []
        for perp, along in (
            (left["perp"], t0),
            (left["perp"], t1),
            (partner["perp"], t1),
            (partner["perp"], t0),
        ):
            point = direction * along + normal * perp
            corners.append((float(point[0]), float(point[1])))
        if min(point[0] for point in corners) < -6 or min(point[1] for point in corners) < -6:
            continue
        if max(point[0] for point in corners) > width + 6 or max(point[1] for point in corners) > height + 6:
            continue
        quad = tuple((int(round(x)), int(round(y))) for x, y in corners)
        xs = [point[0] for point in quad]
        ys = [point[1] for point in quad]
        xyxy = (
            max(0, min(xs)),
            max(0, min(ys)),
            min(width - 1, max(xs)),
            min(height - 1, max(ys)),
        )
        if xyxy[2] - xyxy[0] < 8 or xyxy[3] - xyxy[1] < 8:
            continue
        quads.append((quad, xyxy))
    return quads


def _asphalt_level_frame(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plain = gray[(hsv[:, :, 1] < 32) & (gray > 40) & (gray < 190)]
    if plain.size < 40:
        return 140.0
    return float(np.median(plain))


def _quad_is_green(image: np.ndarray, quad, ratio: float = 0.28) -> bool:
    packed = _quad_crop(image, quad)
    if packed is None:
        return False
    crop, mask = packed
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green = (hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 95) & (hsv[:, :, 1] > 40) & (mask > 0)
    return float(green.sum()) / float(mask.sum()) > ratio


def _quad_crop(image: np.ndarray, quad):
    height, width = image.shape[:2]
    x1 = max(0, min(point[0] for point in quad))
    y1 = max(0, min(point[1] for point in quad))
    x2 = min(width, max(point[0] for point in quad) + 1)
    y2 = min(height, max(point[1] for point in quad) + 1)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = image[y1:y2, x1:x2]
    mask = np.zeros(crop.shape[:2], np.uint8)
    local = np.array([[x - x1, y - y1] for x, y in quad], dtype=np.int32)
    cv2.fillConvexPoly(mask, local, 1)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    if int(mask.sum()) < 20:
        return None
    return crop, mask


def _quad_center(image: np.ndarray, quad) -> tuple[float, float]:
    """Median and spread of the middle of a bay, away from its painted edge."""
    height, width = image.shape[:2]
    points = np.array(quad, np.float32)
    moment = cv2.moments(points.reshape(-1, 1, 2))
    if moment["m00"] == 0:
        return 0.0, 0.0
    center = np.array([moment["m10"] / moment["m00"], moment["m01"] / moment["m00"]])
    shrunk = np.int32(np.round((points - center) * 0.5 + center))
    core = np.zeros((height, width), np.uint8)
    if cv2.contourArea(shrunk) < 20:
        return 0.0, 0.0
    cv2.fillConvexPoly(core, shrunk, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    values = gray[core > 0]
    if values.size < 16:
        return 0.0, 0.0
    return float(np.median(values)), float(values.std())


def _mask_fraction(mask: np.ndarray, quad) -> float:
    """Share of a bay that lands on the vehicle mask."""
    height, width = mask.shape
    filled = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(filled, np.array(quad, np.int32), 1)
    if int(filled.sum()) < 20:
        return 0.0
    return float(mask[filled > 0].mean())


def _poly_iou(quad_a, quad_b) -> float:
    """Overlap of two slanted bays. Their upright boxes overlap even when the stalls do not."""
    area, _ = cv2.intersectConvexConvex(np.array(quad_a, np.float32), np.array(quad_b, np.float32))
    if area <= 0:
        return 0.0
    union = cv2.contourArea(np.array(quad_a, np.float32)) + cv2.contourArea(np.array(quad_b, np.float32)) - area
    if union <= 0:
        return 0.0
    return float(area) / float(union)


def _foliage_fraction(hsv) -> float:
    """Share of pixels that are leaves rather than paint, asphalt, or a car."""
    pixels = np.asarray(hsv).reshape(-1, 3)
    if pixels.size == 0:
        return 1.0
    hue, sat = pixels[:, 0], pixels[:, 1]
    return float(((hue >= 18) & (hue <= 48) & (sat > 70)).mean())


def _core_sample(gray, hsv, probe, cx, cy, width, length):
    """Vehicle fraction, gray spread, and foliage inside a rotated-space bay."""
    half_w, half_l = width * 0.30, length * 0.30
    x1, x2 = int(cx - half_w), int(cx + half_w)
    y1, y2 = int(cy - half_l), int(cy + half_l)
    height, width_px = gray.shape
    if x1 < 2 or y1 < 2 or x2 >= width_px - 2 or y2 >= height - 2:
        return None
    return (
        float(probe[y1:y2, x1:x2].mean()),
        float(gray[y1:y2, x1:x2].std()),
        _foliage_fraction(hsv[y1:y2, x1:x2]),
        float(np.median(gray[y1:y2, x1:x2])),
    )


def _quad_from_upright(inverse, cx, cy, half_w, half_l, shape):
    """Map an axis-aligned bay in the rotated view back onto the photo."""
    height, width = shape[:2]

    def original(x, y):
        return (
            inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2],
            inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2],
        )

    center = original(cx, cy)
    if not (10 < center[0] < width - 10 and 10 < center[1] < height - 10):
        return None
    corners = [original(cx + dx, cy + dy) for dx in (-half_w, half_w) for dy in (-half_l, half_l)]
    order = np.argsort([np.arctan2(y - center[1], x - center[0]) for x, y in corners])
    ordered = [corners[index] for index in order]
    return tuple((int(round(np.clip(x, 0, width - 1))), int(round(np.clip(y, 0, height - 1)))) for x, y in ordered)


def _upright_vehicles(image, peak):
    """Cars whose long axis follows this slant, in a view where that axis is vertical.

    Dark bodies and light roofs are separate blobs. A bush is dropped from the
    hue of its own pixels, and a blob aimed off the slant is a car from the
    other row.
    """
    turned, matrix = _rotate_bound(image, 90 - peak)
    inverse = np.linalg.inv(np.vstack([matrix, [0, 0, 1]]))[:2]
    gray = cv2.cvtColor(turned, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(turned, cv2.COLOR_BGR2HSV)
    # A 960px frame uses the gates below. A smaller overhead shot has smaller
    # cars, so the same pixel floors would drop most of a herringbone row.
    scale = min(image.shape[:2]) / 960.0
    span = 81 if scale >= 0.72 else (max(31, int(min(image.shape[:2]) * 0.1)) | 1)
    if span % 2 == 0:
        span += 1
    area_lo = int(np.clip(520 * scale, 340, 520))
    short_lo = float(np.clip(22 * scale, 16, 22))
    long_lo = float(np.clip(52 * scale, 40, 52))
    delta = gray.astype(np.int16) - cv2.medianBlur(gray, span).astype(np.int16)
    probe = ((delta < -16) | (delta > 12)).astype(np.uint8)
    blobs = []
    for darker, limit in ((True, -24), (False, 18)):
        mask = (delta < limit).astype(np.uint8) if darker else (delta > limit).astype(np.uint8)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(closed)
        for index in range(1, count):
            ys, xs = np.where(labels == index)
            if not area_lo < len(xs) < 9000 or _foliage_fraction(hsv[ys, xs]) > 0.22:
                continue
            sample = np.stack([xs, ys], 1)
            if len(sample) > 1100:
                sample = sample[np.linspace(0, len(sample) - 1, 1100).astype(int)]
            rect = cv2.minAreaRect(sample.astype(np.float32))
            box = cv2.boxPoints(rect)
            edges = []
            for start, end in zip(box, np.roll(box, -1, 0)):
                dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
                edges.append((float(np.hypot(dx, dy)), float(np.degrees(np.arctan2(dy, dx)) % 180)))
            edges.sort(key=lambda edge: -edge[0])
            long, axis = edges[0]
            short = edges[2][0]
            if short < short_lo or long < long_lo or long > 170 or not 1.35 <= long / max(short, 1) <= 3.4:
                continue
            if _angle_delta(axis, 90) > 16:
                continue
            blobs.append([float(rect[0][0]), float(rect[0][1]), short, long, float(len(xs))])
    blobs.sort(key=lambda blob: -blob[4])
    kept = []
    for blob in blobs:
        if any(
            abs(blob[0] - other[0]) < 0.65 * max(blob[2], other[2])
            and abs(blob[1] - other[1]) < 0.5 * max(blob[3], other[3])
            for other in kept
        ):
            continue
        kept.append(blob)
    return kept, inverse, gray, hsv, probe


def _row_pitch_and_slopes(cars):
    """Stall spacing and how much each row drifts, from cars parked side by side."""
    if len(cars) < 4:
        return None
    med_w = float(np.median([car[2] for car in cars]))
    med_l = float(np.median([car[3] for car in cars]))
    pairs = []
    for index, car in enumerate(cars):
        best = None
        for other_index, other in enumerate(cars):
            dx = other[0] - car[0]
            if not (0.85 * med_w <= dx <= 2.4 * med_w):
                continue
            dy = other[1] - car[1]
            if abs(dy) > 0.75 * dx or abs(dy) > 0.6 * med_l:
                continue
            if best is None or dx < best[0]:
                best = (dx, dy, other_index)
        if best is not None:
            pairs.append((index, best[2], best[0], best[1] / best[0]))
    pitch = float(np.median([pair[2] for pair in pairs])) if pairs else med_w * 1.45
    slopes = np.array([pair[3] for pair in pairs]) if pairs else np.array([0.0])
    if slopes.max() - slopes.min() > 0.45:
        centers = [
            float(np.median(slopes[slopes >= np.median(slopes)])),
            float(np.median(slopes[slopes < np.median(slopes)])),
        ]
    else:
        centers = [float(np.median(slopes))]
    return med_w, med_l, pitch, pairs, centers


def _slope_for(car, cars, pairs, centers, pitch):
    """Row drift at this car: the nearest side-by-side pair, or the family slope."""
    best = None
    for left, right, _dx, slope in pairs:
        midpoint_x = (cars[left][0] + cars[right][0]) / 2
        midpoint_y = (cars[left][1] + cars[right][1]) / 2
        dist = abs(car[0] - midpoint_x) + abs(car[1] - midpoint_y)
        if best is None or dist < best[0]:
            best = (dist, slope)
    if best is not None and best[0] < pitch * 5:
        return best[1]
    return min(centers, key=lambda slope: abs(slope))


def _vertical_dividers(image: np.ndarray, peak: float):
    """Dividers of one slant, in a view where that slant stands upright.

    The line angle is measured with y growing downward, and OpenCV rotates
    the other way, so ``peak - 90`` is the turn that makes these strokes
    vertical. Pieces of one divider are joined when they share that column.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    turned, matrix = _rotate_bound(image, peak - 90)
    inverse = np.linalg.inv(np.vstack([matrix, [0, 0, 1]]))[:2]
    pieces = []
    for x1, y1, x2, y2, ang, length in _paint_segments(gray):
        if _angle_delta(ang, peak) > 12 or length < 16:
            continue
        mapped = cv2.transform(np.array([[[x1, y1], [x2, y2]]], np.float32), matrix)[0]
        if abs(float(mapped[0, 0] - mapped[1, 0])) > 8:
            continue
        pieces.append(
            (
                float(mapped[:, 0].mean()),
                float(min(mapped[0, 1], mapped[1, 1])),
                float(max(mapped[0, 1], mapped[1, 1])),
            )
        )
    lines: list[list[float]] = []
    for x, y0, y1 in pieces:
        host = next(
            (
                line
                for line in lines
                if abs(x - line[0]) <= 7 and y0 <= line[2] + 36 and line[1] <= y1 + 36
            ),
            None,
        )
        if host is None:
            lines.append([x, y0, y1, 1.0])
            continue
        host[0] = (host[0] * host[3] + x) / (host[3] + 1)
        host[3] += 1
        host[1] = min(host[1], y0)
        host[2] = max(host[2], y1)
    return lines, turned, inverse


def _ridge_fraction(gray: np.ndarray, x: float, y0: float, y1: float) -> float:
    """Share of a vertical stroke that is brighter than the asphalt beside it."""
    height, width = gray.shape
    column = int(round(x))
    start, end = int(y0), int(y1)
    if end - start < 12 or not (6 <= column < width - 6):
        return 0.0
    hits = 0
    total = 0
    for y in range(max(0, start), min(height, end), 3):
        total += 1
        center = int(gray[y, column - 3:column + 4].max())
        left = float(gray[y, column - 6:column - 3].mean())
        right = float(gray[y, column + 4:column + 7].mean())
        if center >= left + 10 and center >= right + 10 and center >= 150:
            hits += 1
    if total == 0:
        return 0.0
    return hits / float(total)


def _painted_open_bays(image: np.ndarray, peaks: list[float]) -> list[tuple]:
    """Open stalls with a painted line on both long sides.

    A whole empty column has no parked car to bracket it, so a walk that
    starts at a car never enters it. The stall is still the flat asphalt
    between two dividers. Both lines have to run the length of the bay,
    which is what keeps the box off the driving lane in front of the row.
    """
    if not peaks:
        return []
    asphalt = _asphalt_level_frame(image)
    tone_limit = 48 if min(image.shape[:2]) >= 700 else 40
    found = []
    for peak in peaks:
        lines, turned, inverse = _vertical_dividers(image, peak)
        if len(lines) < 4:
            continue
        lines.sort(key=lambda line: line[0])
        gray = cv2.cvtColor(turned, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(turned, cv2.COLOR_BGR2HSV)
        gaps = []
        for index, left in enumerate(lines):
            for right in lines[index + 1:index + 6]:
                dist = right[0] - left[0]
                overlap = min(left[2], right[2]) - max(left[1], right[1])
                if 26 <= dist <= 90 and overlap >= 20:
                    gaps.append(dist)
                    break
        if len(gaps) < 3:
            continue
        pitch = float(np.median(gaps))
        stall_l = float(np.clip(pitch * 2.05, 64, 150))
        for index, left in enumerate(lines):
            partner = None
            partner_dist = 1e9
            for right in lines[index + 1:index + 7]:
                dist = right[0] - left[0]
                if pitch * 0.78 <= dist <= pitch * 1.28 and dist < partner_dist:
                    partner, partner_dist = right, dist
            if partner is None:
                continue
            y_start = min(left[1], partner[1])
            y_end = max(left[2], partner[2])
            if y_end - y_start < stall_l * 0.75:
                continue
            y = y_start + stall_l * 0.5
            while y < y_end - stall_l * 0.25:
                top, bottom = y - stall_l / 2, y + stall_l / 2
                cover_left = (min(left[2], bottom) - max(left[1], top)) / stall_l
                cover_right = (min(partner[2], bottom) - max(partner[1], top)) / stall_l
                sided = (
                    cover_left >= 0.55
                    and cover_right >= 0.55
                    and _ridge_fraction(gray, left[0], top, bottom) >= 0.45
                    and _ridge_fraction(gray, partner[0], top, bottom) >= 0.45
                )
                if sided:
                    cx = (left[0] + partner[0]) / 2
                    half_w = partner_dist * 0.42
                    half_l = stall_l * 0.46
                    x1, x2 = int(cx - half_w * 0.5), int(cx + half_w * 0.5)
                    y1, y2 = int(y - half_l * 0.7), int(y + half_l * 0.7)
                    height, width = gray.shape
                    if 2 < x1 < x2 < width - 2 and 2 < y1 < y2 < height - 2:
                        core = gray[y1:y2, x1:x2]
                        tone = float(np.median(core))
                        spread = float(core.std())
                        if (
                            spread <= 16
                            and abs(tone - asphalt) <= tone_limit
                            and _foliage_fraction(hsv[y1:y2, x1:x2]) <= 0.08
                        ):
                            quad = _quad_from_upright(inverse, cx, y, half_w, half_l, image.shape)
                            if quad is not None:
                                found.append((quad, True))
                y += stall_l * 0.55
    return found


def _slant_car_slots(image: np.ndarray, peaks: list[float]) -> list[Slot]:
    """One stall per vehicle, plus the open bays of that row.

    The photo is turned so each slant stands vertical. A car is then a tall
    blob, and the open stall between two cars of the row is the flat asphalt
    one step along it. A column of open stalls has no car to start from, so
    those are read from the painted sides instead. Neither walk is allowed
    to leave a box in the driving lane.
    """
    if not peaks:
        return []
    found: list[tuple] = []
    for peak in peaks:
        cars, inverse, gray, hsv, probe = _upright_vehicles(image, peak)
        fitted = _row_pitch_and_slopes(cars)
        if fitted is None:
            continue
        med_w, med_l, pitch, pairs, centers = fitted
        asphalt = _asphalt_level_frame(image)
        stall_l = float(np.clip(med_l, 40, med_l * 1.02))
        stall_w = float(np.clip(min(med_w * 1.08, pitch * 0.78), med_w * 0.75, pitch * 0.86))
        for car in cars:
            stats = _core_sample(gray, hsv, probe, car[0], car[1], stall_w, stall_l)
            if stats is None or stats[2] > 0.12 or stats[0] < 0.16:
                continue
            quad = _quad_from_upright(inverse, car[0], car[1], stall_w / 2, stall_l / 2, image.shape)
            if quad is not None:
                found.append((quad, False))
            slope = _slope_for(car, cars, pairs, centers, pitch)
            for direction in (1, -1):
                cx = car[0] + direction * pitch
                cy = car[1] + direction * pitch * slope
                if any(abs(cx - other[0]) < pitch * 0.42 and abs(cy - other[1]) < med_l * 0.42 for other in cars):
                    continue
                stats = _core_sample(gray, hsv, probe, cx, cy, stall_w, stall_l)
                if stats is None or stats[2] > 0.08 or not (stats[0] <= 0.08 and stats[1] <= 16):
                    continue
                # On a smaller frame the lane between rows is a bright flat patch.
                # An open stall stays close to the asphalt; that patch does not.
                if min(image.shape[:2]) < 700 and abs(stats[3] - asphalt) > 18:
                    continue
                wide = _core_sample(gray, hsv, probe, cx, cy, stall_w * 1.25, stall_l * 1.05)
                if wide is None or wide[0] > 0.14 or wide[2] > 0.1:
                    continue
                if min(image.shape[:2]) < 700 and abs(wide[3] - asphalt) > 18:
                    continue
                best_residual = 1e9
                for other in cars:
                    along = direction * (other[0] - car[0])
                    if not (1.25 * pitch <= along <= 4.0 * pitch):
                        continue
                    residual = abs((other[1] - car[1]) - (other[0] - car[0]) * slope)
                    best_residual = min(best_residual, residual)
                if best_residual > 0.75 * med_l:
                    continue
                quad = _quad_from_upright(inverse, cx, cy, stall_w / 2, stall_l / 2, image.shape)
                if quad is not None:
                    found.append((quad, True))
        # A car the blob detector split into scraps still sits between two
        # stalls we did see. Fill only that gap, so the walk cannot leave the row.
        anchors = []
        for car in cars:
            stats = _core_sample(gray, hsv, probe, car[0], car[1], stall_w, stall_l)
            if stats is None or stats[2] > 0.12 or stats[0] < 0.16:
                continue
            anchors.append(car)
        for car in anchors:
            slope = _slope_for(car, cars, pairs, centers, pitch)
            partner = None
            for other in anchors:
                dx = other[0] - car[0]
                if dx <= pitch * 1.4:
                    continue
                residual = abs((other[1] - car[1]) - dx * slope)
                if residual > 0.4 * med_l:
                    continue
                if partner is None or dx < partner[0]:
                    partner = (dx, other)
            if partner is None or partner[0] > pitch * 3.4:
                continue
            missing = int(round(partner[0] / pitch)) - 1
            if missing not in (1, 2):
                continue
            other = partner[1]
            for step in range(1, missing + 1):
                cx = car[0] + partner[0] * step / (missing + 1)
                cy = car[1] + (other[1] - car[1]) * step / (missing + 1)
                if any(abs(cx - anchor[0]) < pitch * 0.4 and abs(cy - anchor[1]) < med_l * 0.4 for anchor in anchors):
                    continue
                stats = _core_sample(gray, hsv, probe, cx, cy, stall_w * 0.9, stall_l * 0.85)
                if stats is None or stats[2] > 0.16:
                    continue
                fraction, spread, leaves, tone = stats
                if fraction >= 0.5 and spread >= 20:
                    empty = False
                elif (
                    fraction <= 0.10
                    and spread <= 18
                    and leaves <= 0.08
                    and (min(image.shape[:2]) >= 700 or abs(tone - asphalt) <= 18)
                ):
                    empty = True
                else:
                    continue
                quad = _quad_from_upright(inverse, cx, cy, stall_w / 2, stall_l / 2, image.shape)
                if quad is not None:
                    found.append((quad, empty))
    found.extend(_painted_open_bays(image, peaks))
    found.sort(key=lambda item: item[1])
    kept: list[tuple] = []
    for quad, empty in found:
        limit = 0.06 if empty else 0.18
        if any(_poly_iou(quad, other[0]) > limit for other in kept):
            continue
        kept.append((quad, empty))
    if len(kept) < _LINED_MIN_BAYS:
        return []
    slots = []
    for quad, empty in kept:
        xs = [point[0] for point in quad]
        ys = [point[1] for point in quad]
        slots.append(
            Slot(
                status="empty" if empty else "occupied",
                confidence=0.8 if empty else 0.84,
                xyxy=(min(xs), min(ys), max(xs), max(ys)),
                quad=quad,
            )
        )
    return slots



def _slots_along_slant(image_bgr: np.ndarray) -> list[Slot]:
    """Read bays whose painted dividers run diagonally across the frame.

    Vehicles are boxed along the row they belong to. The open stall between
    two of those cars is marked empty. Painted ticks are only used when the
    vehicles do not already show the rows.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    peaks = _slant_peaks(gray)
    if not peaks:
        return []
    cars = _slant_car_slots(image_bgr, peaks)
    if len(cars) >= _LINED_MIN_BAYS:
        return cars
    found = []
    for degree in peaks:
        found.extend(_quads_for_family(image_bgr, degree))
    asphalt = _asphalt_level_frame(image_bgr)
    scored = []
    for quad, xyxy in found:
        if _quad_is_green(image_bgr, quad):
            continue
        median, spread = _quad_center(image_bgr, quad)
        # The lane is flat asphalt. A stall we can trust has a vehicle in it:
        # darker, lighter, or simply less uniform than the pavement.
        if spread < 14:
            continue
        if abs(median - asphalt) <= 22 and spread <= 28:
            continue
        scored.append((quad, xyxy, spread))
    scored.sort(key=lambda item: item[2], reverse=True)
    kept = []
    for quad, xyxy, spread in scored:
        if any(_iou(xyxy, other[1]) > 0.22 for other in kept):
            continue
        kept.append((quad, xyxy, spread))
    if len(kept) < _LINED_MIN_BAYS:
        return []
    return [
        Slot(status="occupied", confidence=0.8, xyxy=xyxy, quad=quad)
        for quad, xyxy, _spread in kept
    ]


def read_lined(image_bgr: np.ndarray) -> tuple[list[Slot], np.ndarray]:
    """Return bays traced from painted dividers, and the image they sit on."""
    slanted = _slots_along_slant(image_bgr)
    if len(slanted) >= _LINED_MIN_BAYS:
        return slanted, image_bgr
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    vertical = _divider_weight(gray, vertical=True)
    horizontal = _divider_weight(gray, vertical=False)
    if horizontal > vertical * 1.25 and horizontal > 200:
        turned = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
        slots = _slots_upright(turned)
        if len(slots) >= _LINED_MIN_BAYS:
            height = image_bgr.shape[0]
            mapped = [
                Slot(slot.status, slot.confidence, _unmap_clockwise(slot.xyxy, height))
                for slot in slots
            ]
            return mapped, image_bgr
    return _slots_upright(image_bgr), image_bgr


MODEL_PATH = ROOT / "models" / "pklot-yolov8n.pt"
# Trained at 960. Lower confidence keeps faint bays; the model is already strict.
_CONFIDENCE = 0.30
_IMAGE_SIZE = 960
_MODEL_MIN_BAYS = 4


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

    A surveillance-style lot uses the PKLot model. A straight top-down lot
    is read from the painted stall dividers when that finds more bays than
    the model, which only catches a handful of cars from directly above.
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

    lined_slots, lined_image = read_lined(image_bgr)
    if len(lined_slots) > len(best_slots) and len(lined_slots) >= _LINED_MIN_BAYS:
        return LotReading(slots=lined_slots, same_camera=False, aligned_bgr=lined_image, source="lines")
    if len(best_slots) >= _MODEL_MIN_BAYS:
        return LotReading(slots=best_slots, same_camera=False, aligned_bgr=best_image, source="model")

    if len(lined_slots) >= _MODEL_MIN_BAYS:
        return LotReading(slots=lined_slots, same_camera=False, aligned_bgr=lined_image, source="lines")
    return LotReading(slots=[], same_camera=False, aligned_bgr=image_bgr, source="lines")


def list_samples() -> list[Path]:
    if not SAMPLES.is_dir():
        return []
    return sorted(SAMPLES.glob("*.jpg")) + sorted(SAMPLES.glob("*.png"))


@dataclass(frozen=True)
class OpenAverage:
    fraction: float
    open_bays: float
    frames: int


@lru_cache(maxsize=1)
def open_bay_average() -> OpenAverage:
    """Mean share of bays left open across the saved frames of this camera."""
    paths = list_samples()
    if not paths:
        return OpenAverage(_FALLBACK_OPEN_FRACTION, 46.0, 0)
    plate = load_reference(SAMPLES)
    fractions: list[float] = []
    opens: list[int] = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        occupied, empty = count_slots(read_lot(image, plate).slots)
        total = occupied + empty
        if total == 0:
            continue
        fractions.append(empty / total)
        opens.append(empty)
    if not fractions:
        return OpenAverage(_FALLBACK_OPEN_FRACTION, 46.0, 0)
    return OpenAverage(
        fraction=sum(fractions) / len(fractions),
        open_bays=sum(opens) / len(opens),
        frames=len(fractions),
    )


def band_counts(total: int) -> tuple[int, int]:
    """Open-bay counts for full (at or under) and free (at or over) on a lot of this size."""
    average = open_bay_average().fraction
    full_at = int(average * 0.5 * total)
    free_line = average * 2 * total
    free_at = int(free_line) if abs(free_line - round(free_line)) < 1e-6 else int(free_line) + 1
    return full_at, free_at


def read_rgb(file) -> np.ndarray:
    return np.array(Image.open(file).convert("RGB"))


def level_class(level: str) -> str:
    return {"Low": "low", "Moderate": "moderate", "High": "high"}.get(level, "none")


def cue_for(level: str) -> str | None:
    """Sound for a congestion level. None when there is nothing to announce."""
    return {"Low": "clear", "Moderate": "warning", "High": "alarm"}.get(level)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(_SAMPLE_RATE * seconds), dtype=np.float64)


def _beep(frequency: float, seconds: float, volume: float, decay: float) -> np.ndarray:
    count = max(1, int(_SAMPLE_RATE * seconds))
    time = np.arange(count) / _SAMPLE_RATE
    tone = np.sin(2 * np.pi * frequency * time)
    tone += 0.22 * np.sin(2 * np.pi * frequency * 2 * time)
    edge = min(int(0.012 * _SAMPLE_RATE), count // 4)
    envelope = np.ones(count)
    if edge:
        envelope[:edge] = np.linspace(0.0, 1.0, edge)
        envelope[-edge:] = np.linspace(1.0, 0.0, edge)
    envelope *= np.exp(-decay * time / max(seconds, 0.05))
    return tone * envelope * volume


def _wav(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(_SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


@lru_cache(maxsize=3)
def cue_audio(cue: str) -> bytes:
    """Short cue for an open lot, a warning for a busy one, or a 3 second alarm."""
    if cue == "alarm":
        pattern = np.concatenate(
            [
                _beep(880, 0.16, 0.55, decay=0.4),
                _silence(0.06),
                _beep(1175, 0.16, 0.55, decay=0.4),
                _silence(0.12),
            ]
        )
        target = int(3.0 * _SAMPLE_RATE)
        repeats = int(np.ceil(target / len(pattern)))
        samples = np.tile(pattern, repeats)[:target]
        fade = int(0.08 * _SAMPLE_RATE)
        samples[-fade:] *= np.linspace(1.0, 0.0, fade)
    elif cue == "warning":
        samples = np.concatenate(
            [
                _beep(784, 0.14, 0.5, decay=2.2),
                _silence(0.04),
                _beep(523, 0.28, 0.5, decay=2.4),
            ]
        )
    elif cue == "clear":
        samples = np.concatenate(
            [
                _beep(523, 0.12, 0.45, decay=2.0),
                _beep(659, 0.12, 0.45, decay=2.0),
                _beep(784, 0.14, 0.48, decay=2.0),
                _beep(1047, 0.32, 0.42, decay=2.6),
            ]
        )
    else:
        raise ValueError(f"Unknown cue {cue}")
    return _wav(samples)


def _play_cue(cue: str) -> None:
    """Start the cue for this result, including a repeat of the same scenario."""
    st.session_state.cue_play = st.session_state.get("cue_play", 0) + 1
    encoded = base64.b64encode(cue_audio(cue)).decode("ascii")
    # A new id each time forces the browser to start the clip again.
    # The same alarm file would otherwise stay silent on the next full lot.
    st.markdown(
        f'<audio autoplay playsinline id="parkvision-cue-{st.session_state.cue_play}" '
        f'src="data:audio/wav;base64,{encoded}"></audio>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def reference():
    return load_reference(SAMPLES)


def _frame_label(name: str) -> str:
    """Short chip label for a saved frame, such as ``01`` from ``lot-01.jpg``."""
    stem = Path(name).stem
    if stem.startswith("lot-"):
        return stem[4:]
    return stem


def _mosaic_html(slots: list[Slot]) -> str:
    """One cell per bay, ordered from the top of the frame downward."""

    def center(slot: Slot) -> tuple[float, float]:
        if slot.quad:
            xs = [point[0] for point in slot.quad]
            ys = [point[1] for point in slot.quad]
            return (sum(ys) / len(ys), sum(xs) / len(xs))
        x1, y1, x2, y2 = slot.xyxy
        return ((y1 + y2) / 2, (x1 + x2) / 2)

    cells = []
    for slot in sorted(slots, key=center):
        kind = "open" if slot.status == "empty" else "taken"
        title = "Empty" if slot.status == "empty" else "Occupied"
        cells.append(f'<span class="cell {kind}" title="{title}"></span>')
    return (
        '<div class="mosaic" aria-label="Bays from the top of the frame">'
        + "".join(cells)
        + "</div>"
    )


def _rail_html(insights: ParkingInsights, slots: list[Slot], full_at: int, free_at: int, usual_open: float) -> str:
    """The call, the counts, and where this frame sits between full and free."""
    tone = level_class(insights.level)
    color = {"low": "#3ddc84", "moderate": "#e0a23a", "high": "#ff5d5d"}.get(tone, "#8ea0b5")
    occupancy = max(0.0, min(100.0, insights.occupancy_pct))
    span = max(insights.total, 1)
    now = min(100.0, 100.0 * insights.available / span)
    full_mark = min(100.0, 100.0 * full_at / span)
    free_mark = min(100.0, 100.0 * free_at / span)
    return f"""
    <div class="rail {tone}">
      <div class="rail-top">
        <div class="ring" style="background: conic-gradient({color} {occupancy:.1f}%, rgba(232,238,246,0.08) 0);">
          <div class="ring-hole">
            <strong>{insights.occupancy_pct:.1f}%</strong>
            <span>full</span>
          </div>
        </div>
        <div class="verdict">
          <p class="chip">{insights.level}</p>
          <h2>{insights.available} open</h2>
          <p class="say">{insights.recommendation}</p>
        </div>
      </div>
      <div class="stats">
        <div><span>Total</span><strong>{insights.total}</strong></div>
        <div><span>Occupied</span><strong>{insights.occupied}</strong></div>
        <div><span>Available</span><strong>{insights.available}</strong></div>
        <div><span>Occupancy</span><strong>{insights.occupancy_pct:.1f}%</strong></div>
      </div>
      {_mosaic_html(slots)}
      <p class="mosaic-note">Each square is one bay, from the top of the frame.</p>
      <div class="band">
        <div class="band-track">
          <div class="band-now" style="left:{now:.1f}%"></div>
          <div class="band-tick" style="left:{full_mark:.1f}%"></div>
          <div class="band-tick free" style="left:{free_mark:.1f}%"></div>
        </div>
        <div class="band-labels">
          <span>Full at {full_at}</span>
          <span>{insights.available} open now</span>
          <span>Free at {free_at}</span>
        </div>
      </div>
      <p class="usual">A normal frame of the sample camera leaves {usual_open:.0f} bays open. The sound for this band plays on its own.</p>
    </div>
    """


_PAGE_CSS = """
<style>
  html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #090c11;
    color: #e8eef6;
    font-family: Inter, "Segoe UI", sans-serif;
  }
  header[data-testid="stHeader"],
  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  [data-testid="stStatusWidget"] {
    display: none !important;
    height: 0 !important;
  }
  #MainMenu, footer { visibility: hidden; height: 0; }
  .block-container {
    max-width: 1180px;
    padding-top: 1.15rem;
    padding-bottom: 2.6rem;
  }
  .brand { display: flex; align-items: center; gap: 0.85rem; margin: 0 0 0.35rem; }
  .brand img { width: 3.1rem; height: 3.1rem; border-radius: 0.8rem; }
  .brand .eyebrow {
    margin: 0;
    color: #8fb4ff;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-size: 0.68rem;
  }
  .brand h1 { margin: 0; font-size: 1.55rem; letter-spacing: -0.03em; line-height: 1.1; }
  .lede { color: #9aabbe; margin: 0 0 0.85rem; max-width: 40rem; font-size: 0.95rem; }
  .st-key-stage, .st-key-railbox {
    background: #12181f;
    border: 1px solid rgba(232, 238, 246, 0.08);
    border-radius: 18px;
    padding: 0.85rem 0.9rem 0.35rem;
  }
  .view-label {
    margin: 0 0 0.45rem;
    color: #8ea0b5;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  [data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid rgba(232, 238, 246, 0.08);
  }
  .legend { display: flex; gap: 0.9rem; color: #c5d0de; font-size: 0.82rem; margin: 0.15rem 0 0.2rem; }
  .swatch {
    display: inline-block; width: 0.7rem; height: 0.7rem;
    border-radius: 2px; margin-right: 0.35rem; vertical-align: -1px;
  }
  .rail { display: flex; flex-direction: column; gap: 0.85rem; }
  .rail-top { display: flex; gap: 0.9rem; align-items: center; }
  .ring {
    width: 148px; height: 148px; border-radius: 50%;
    display: grid; place-items: center; flex: none;
  }
  .ring-hole {
    width: 108px; height: 108px; border-radius: 50%;
    background: #12181f;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
  }
  .ring-hole strong {
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 1.35rem; letter-spacing: -0.04em; line-height: 1;
  }
  .ring-hole span { color: #8ea0b5; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; }
  .chip {
    display: inline-block; margin: 0 0 0.25rem;
    padding: 0.12rem 0.45rem; border-radius: 999px;
    font-size: 0.68rem; letter-spacing: 0.12em; text-transform: uppercase;
    border: 1px solid transparent;
  }
  .low .chip { color: #3ddc84; border-color: rgba(61, 220, 132, 0.45); background: rgba(61, 220, 132, 0.1); }
  .moderate .chip { color: #e0a23a; border-color: rgba(224, 162, 58, 0.45); background: rgba(224, 162, 58, 0.1); }
  .high .chip { color: #ff5d5d; border-color: rgba(255, 93, 93, 0.5); background: rgba(255, 93, 93, 0.1); }
  .none .chip { color: #8ea0b5; border-color: rgba(142, 160, 181, 0.4); }
  .verdict h2 {
    margin: 0; font-size: 1.85rem; letter-spacing: -0.04em; line-height: 1;
    font-family: "JetBrains Mono", ui-monospace, monospace;
  }
  .verdict .say { margin: 0.4rem 0 0; color: #d5deea; font-size: 0.95rem; }
  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.45rem; }
  .stats div {
    background: rgba(232, 238, 246, 0.04);
    border-radius: 10px; padding: 0.45rem 0.5rem 0.4rem;
  }
  .stats span { display: block; color: #8ea0b5; font-size: 0.68rem; letter-spacing: 0.06em; text-transform: uppercase; }
  .stats strong {
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 1.15rem; font-weight: 560;
  }
  .mosaic { display: flex; flex-wrap: wrap; gap: 3px; }
  .cell { width: 12px; height: 16px; border-radius: 2px; display: inline-block; }
  .cell.open { background: #3ddc84; }
  .cell.taken { background: #ff5d5d; }
  .mosaic-note, .usual, .fineprint { color: #8ea0b5; font-size: 0.8rem; margin: 0; }
  .band-track {
    position: relative; height: 6px; border-radius: 999px;
    background: rgba(232, 238, 246, 0.1); margin: 0.2rem 0 0.35rem;
  }
  .band-now {
    position: absolute; top: -4px; width: 14px; height: 14px; border-radius: 50%;
    background: #e8eef6; transform: translateX(-50%);
    box-shadow: 0 0 0 3px rgba(232, 238, 246, 0.16);
  }
  .low .band-now { background: #3ddc84; }
  .moderate .band-now { background: #e0a23a; }
  .high .band-now { background: #ff5d5d; }
  .band-tick {
    position: absolute; top: -2px; width: 2px; height: 10px;
    background: rgba(232, 238, 246, 0.45); transform: translateX(-50%);
  }
  .band-labels { display: flex; justify-content: space-between; gap: 0.4rem; color: #8ea0b5; font-size: 0.72rem; }
  .empty-stage {
    border: 1px dashed rgba(143, 180, 255, 0.35);
    border-radius: 18px; padding: 2.2rem 1.4rem; text-align: center;
    background: #10161e;
  }
  .empty-stage h2 { margin: 0.4rem 0 0.3rem; font-size: 1.25rem; }
  .empty-stage p { margin: 0 auto; max-width: 34rem; color: #9aabbe; }
  [data-testid="stFileUploaderDropzone"] {
    background: #10161e;
    border: 1px dashed rgba(143, 180, 255, 0.45);
    border-radius: 16px;
  }
  .cue { height: 0; overflow: hidden; }
  @media (max-width: 720px) {
    .rail-top { flex-direction: column; align-items: flex-start; }
    .stats { grid-template-columns: repeat(2, 1fr); }
    .brand h1 { font-size: 1.3rem; }
    .verdict h2 { font-size: 1.55rem; }
  }
</style>
"""


# The mark painted in the header. Used when assets/logo.png is not next to
# the app, which is what a Streamlit Cloud deploy does if that folder is absent.
_LOGO_PNG = (
    "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"
)


def _logo_bytes() -> bytes:
    """The header mark, from disk when the file is deployed and otherwise from the app."""
    path = ROOT / "assets" / "logo.png"
    if path.is_file():
        return path.read_bytes()
    return base64.b64decode(_LOGO_PNG)


def main() -> None:
    logo = _logo_bytes()
    icon_path = ROOT / "assets" / "logo.png"
    st.set_page_config(
        page_title="ParkVision AI",
        page_icon=str(icon_path) if icon_path.is_file() else Image.open(io.BytesIO(logo)),
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_PAGE_CSS, unsafe_allow_html=True)

    mark = base64.b64encode(logo).decode("ascii")
    st.markdown(
        f"""
        <div class="brand">
          <img src="data:image/png;base64,{mark}" alt="ParkVision">
          <div>
            <p class="eyebrow">Overhead lot reading</p>
            <h1>ParkVision AI</h1>
          </div>
        </div>
        <p class="lede">
          An overhead shot of a lined lot. Empty bays are green, occupied bays
          are red, and the call tells a driver whether to pull in.
        </p>
        """,
        unsafe_allow_html=True,
    )

    samples = list_samples()
    source_options = ["Sample lot", "Upload"] if samples else ["Upload"]
    with st.container(key="controls"):
        source = st.segmented_control(
            "Frame",
            source_options,
            default=source_options[0],
            selection_mode="single",
            key="source_mode",
        )
        upload = None
        sample_path = None
        if source == "Upload":
            upload = st.file_uploader(
                "Overhead photo",
                type=["jpg", "jpeg", "png", "webp"],
                help="An overhead photo of a lined parking lot. It does not have to be the sample camera.",
            )
        elif samples and source == "Sample lot":
            labels = [path.name for path in samples]
            default = "lot-01.jpg" if "lot-01.jpg" in labels else labels[0]
            choice = st.pills(
                "Saved frames",
                labels,
                default=default,
                format_func=_frame_label,
                selection_mode="single",
                key="frame",
            )
            if choice:
                sample_path = samples[labels.index(choice)]

    image = None
    if upload is not None:
        image = read_rgb(upload)
    elif sample_path is not None:
        image = read_rgb(sample_path)

    if image is None:
        if source != "Upload":
            st.markdown(
                """
                <div class="empty-stage">
                  <h2>Choose a frame</h2>
                  <p>Pick a saved lot. The chips above are the frames from this camera.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.stop()

    with st.spinner("Reading this frame…"):
        plate = reference()
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        reading = read_lot(bgr, plate)
        if not reading.slots:
            reading = read_overhead(bgr)

    if not reading.slots:
        st.image(image, width="stretch")
        st.markdown(
            """
            <div class="empty-stage">
              <h2>No bays in this photo</h2>
              <p>Use an overhead shot where the stall lines are visible, or pick a saved frame.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    shown = cv2.cvtColor(reading.aligned_bgr, cv2.COLOR_BGR2RGB)
    occupied, empty = count_slots(reading.slots)
    insights = summarize(occupied, empty)
    annotated = annotate(shown, reading.slots)
    usual = open_bay_average()
    full_at, free_at = band_counts(insights.total)

    stage, rail = st.columns([1.42, 0.78], gap="medium")
    with stage:
        with st.container(key="stage"):
            view = st.segmented_control(
                "View",
                ["Split", "Overlay", "Camera"],
                default="Split",
                selection_mode="single",
                key="view_mode",
            )
            if view is None:
                view = "Split"
            st.markdown(
                """
                <div class="legend">
                  <span><i class="swatch" style="background:#3ddc84"></i>Empty</span>
                  <span><i class="swatch" style="background:#ff5d5d"></i>Occupied</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if view == "Split":
                camera, overlay = st.columns(2, gap="small")
                with camera:
                    st.markdown('<p class="view-label">Camera</p>', unsafe_allow_html=True)
                    st.image(shown, width="stretch")
                with overlay:
                    st.markdown('<p class="view-label">Overlay</p>', unsafe_allow_html=True)
                    st.image(annotated, width="stretch")
            elif view == "Camera":
                st.markdown('<p class="view-label">Camera</p>', unsafe_allow_html=True)
                st.image(shown, width="stretch")
            else:
                st.markdown('<p class="view-label">Overlay</p>', unsafe_allow_html=True)
                st.image(annotated, width="stretch")
    with rail:
        with st.container(key="railbox"):
            st.markdown(
                _rail_html(insights, reading.slots, full_at, free_at, usual.open_bays),
                unsafe_allow_html=True,
            )

    if reading.same_camera:
        note = (
            "This frame matched the sample camera, so the bays come from that view's map. "
            "The empty plate is samples/lot-11.jpg (Almeida et al., 2015, CC BY 4.0)."
        )
    elif reading.source == "lines":
        note = "These bays were traced from the painted lines. Empty asphalt is green. A bay with a vehicle in it is red."
    else:
        note = (
            "These bays were found in the photo itself, so the shot does not have to be the sample camera. "
            "The reader is a YOLOv8n model trained on PKLot."
        )
    st.markdown(f'<p class="fineprint">{note}</p>', unsafe_allow_html=True)

    with st.expander("How the call is made"):
        st.markdown(
            """
            A photo of the sample camera uses a fixed grid of its lined rows.
            Each bay is compared with an empty plate of the same lot. A poor
            match means the bay is occupied. A bay that still roughly matches,
            but is much busier than the empty asphalt, is counted too.

            Any other overhead shot is read from the painted stall lines when
            they are clear, and otherwise by a parking-space model trained on
            lined lots. On a diagonal or herringbone lot, each stall is boxed
            along its row. An open stall is marked when it sits between two
            cars, or when both painted sides are visible and the middle is
            bare asphalt.

            The average number of open bays is taken across the saved frames.
            Twice that average, scaled to this lot, is free and plays a short
            chime. Half of it is full and plays a 3 second alarm. Anywhere
            between is busy and plays a warning. The sound starts on its own,
            including when the next lot falls in the same band.
            """
        )

    cue = cue_for(insights.level)
    if cue is not None:
        _play_cue(cue)


if __name__ == "__main__":
    main()

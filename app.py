"""ParkVision AI.

One Streamlit file. Upload an overhead shot of a lined parking lot, or open a
sample frame. Bays that match the sample camera use that camera's map. Any
other overhead shot is read from a parking-space model, or from the painted
stall lines when the shot is straight down.
"""

from __future__ import annotations

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


# Row bands (top, bottom) as fractions of the frame. They sit on the lined
# asphalt rows and skip the driving aisles and the tree-covered lower court.
ROW_BANDS: tuple[tuple[float, float], ...] = (
    (0.000, 0.070),
    (0.075, 0.150),
    (0.160, 0.245),
    (0.255, 0.340),
    (0.360, 0.430),
    (0.455, 0.540),
    (0.555, 0.650),
)
# Stop before the drive lane on the right. That asphalt matches an empty
# bay and was being reported as free space.
X_START = 0.00
X_STOP = 0.80
X_STEP = 0.05

# Below this normalised correlation the bay is occupied.
OCCUPIED_BELOW = 0.55
# A bay that still resembles the plate, but is much busier, is a partial vehicle.
TEXTURE_RATIO = 4.5
TEXTURE_MARGIN = 0.23

REFERENCE_NAME = "lot-11.jpg"
_ROW_COUNT = len(ROW_BANDS)


@dataclass(frozen=True)
class LotReading:
    slots: list[Slot]
    same_camera: bool
    aligned_bgr: np.ndarray
    # "camera" uses the saved sample map, "model" the PKLot detector,
    # "lines" the painted dividers in a top-down shot.
    source: str = "camera"


def slot_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Pixel boxes for every lined bay, left to right, top to bottom."""
    boxes: list[tuple[int, int, int, int]] = []
    x = X_START
    edges: list[float] = []
    while x < X_STOP - 1e-6:
        edges.append(x)
        x = round(x + X_STEP, 5)
    edges.append(X_STOP)
    for y1, y2 in ROW_BANDS:
        top = int(round(y1 * height))
        bottom = int(round(y2 * height))
        for left, right in zip(edges, edges[1:]):
            x1 = int(round(left * width))
            x2 = int(round(right * width))
            if x2 - x1 >= 2 and bottom - top >= 2:
                boxes.append((x1, top, x2, bottom))
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
    soft_below = min(0.85, occupied_below + TEXTURE_MARGIN)
    slots: list[Slot] = []
    for box in boxes:
        x1, y1, x2, y2 = box
        bay = image_bgr[y1:y2, x1:x2]
        if bay.size == 0:
            continue
        covered = np.count_nonzero(cv2.cvtColor(bay, cv2.COLOR_BGR2GRAY) > 8) / float(bay.shape[0] * bay.shape[1])
        if covered < 0.65:
            continue
        match = _correlation(image_bgr, reference_bgr, box)
        busy = _texture_ratio(image_bgr, reference_bgr, box)
        occupied = match < occupied_below or (match < soft_below and busy > TEXTURE_RATIO)
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


def read_lined(image_bgr: np.ndarray) -> tuple[list[Slot], np.ndarray]:
    """Return bays traced from painted dividers, and the image they sit on."""
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
    if len(best_slots) >= _MODEL_MIN_BAYS:
        return LotReading(slots=best_slots, same_camera=False, aligned_bgr=best_image, source="model")

    lined_slots, lined_image = read_lined(image_bgr)
    if len(lined_slots) >= _MODEL_MIN_BAYS:
        return LotReading(slots=lined_slots, same_camera=False, aligned_bgr=lined_image, source="lines")
    return LotReading(slots=[], same_camera=False, aligned_bgr=image_bgr, source="lines")


def list_samples() -> list[Path]:
    if not SAMPLES.is_dir():
        return []
    return sorted(SAMPLES.glob("*.jpg")) + sorted(SAMPLES.glob("*.png"))


def read_rgb(file) -> np.ndarray:
    return np.array(Image.open(file).convert("RGB"))


def level_class(level: str) -> str:
    return {"Low": "low", "Moderate": "moderate", "High": "high"}.get(level, "none")


@st.cache_resource(show_spinner=False)
def reference():
    return load_reference(SAMPLES)


def main() -> None:
    st.set_page_config(
        page_title="ParkVision AI",
        page_icon="P",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
          .hero { margin: 0.2rem 0 1.2rem; }
          .eyebrow {
            color: #8fb4ff;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-size: 0.75rem;
            margin: 0 0 0.35rem;
          }
          .hero h1 { margin: 0; font-size: 2.1rem; line-height: 1.1; }
          .hero p { color: #b7c3d4; max-width: 46rem; margin: 0.55rem 0 0; }
          .banner {
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-top: 0.4rem;
          }
          .banner h3 { margin: 0 0 0.25rem; font-size: 1.05rem; }
          .banner p { margin: 0; }
          .low { background: #123524; border: 1px solid #1f7a45; }
          .moderate { background: #3a2d12; border: 1px solid #a9781a; }
          .high { background: #3a1717; border: 1px solid #a33b3b; }
          .none { background: #1c2430; border: 1px solid #314056; }
          .legend { display: flex; gap: 1rem; margin: 0.3rem 0 0.8rem; color: #c5d0de; }
          .swatch {
            display: inline-block; width: 0.75rem; height: 0.75rem;
            border-radius: 2px; margin-right: 0.35rem; vertical-align: -1px;
          }
          .note { color: #9aabbe; font-size: 0.92rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
          <p class="eyebrow">UrbanFlow AI · Smart city parking</p>
          <h1>ParkVision AI</h1>
          <p>
            Overhead shots of a lined parking lot. Each bay turns green when it
            is empty and red when a vehicle is in it, then the lot is scored so
            a driver knows whether to pull in.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    samples = list_samples()

    with st.sidebar:
        st.header("Frame")
        source = st.radio(
            "Image source",
            ["Sample lot", "Upload a photo"],
            index=0 if samples else 1,
        )
        upload = None
        sample_path = None
        if source == "Upload a photo":
            upload = st.file_uploader(
                "Overhead shot",
                type=["jpg", "jpeg", "png", "webp"],
                help="An overhead photo of a lined parking lot. It does not have to be the sample camera.",
            )
            st.caption("A shot of this sample lot uses the saved bay map. Any other overhead lot is read from the bays in the photo.")
        elif samples:
            labels = [path.name for path in samples]
            default = labels.index("lot-01.jpg") if "lot-01.jpg" in labels else 0
            choice = st.selectbox("Sample frame", labels, index=default)
            sample_path = samples[labels.index(choice)]
        else:
            st.info("No sample frames are stored in samples/.")

        st.header("Bays")
        occupied_below = st.slider(
            "Match sensitivity",
            min_value=0.35,
            max_value=0.75,
            value=OCCUPIED_BELOW,
            step=0.05,
            help="Used when the photo matches the sample camera. Higher values mark a bay occupied on a weaker mismatch with the empty plate.",
        )

        with st.expander("How a recommendation is made"):
            st.markdown(
                """
                A photo of the sample camera uses a fixed grid of its lined rows.
                Each of those bays is compared with an empty plate of the same lot.
                A poor match means the bay is occupied. A bay that still
                roughly matches, but is much busier than the empty asphalt,
                is counted too.

                Any other overhead shot is read by a parking-space model trained
                on lined lots. Each bay it finds is marked empty or occupied.
                A straight top-down photo is read from the painted stall lines.

                Occupancy is occupied bays divided by every lined bay.

                - **Low** — under 40%. Slots available, proceed.
                - **Moderate** — 40% to 75%. Proceed, and take the nearest empty bay.
                - **High** — over 75%. Parking full, try another area.

                On the sample camera, the tree-covered lower court is outside the
                lined grid, so those cars are not counted.
                """
            )

    image = None
    if upload is not None:
        image = read_rgb(upload)
    elif sample_path is not None and source == "Sample lot":
        image = read_rgb(sample_path)

    if image is None:
        st.info("Choose a sample frame, or upload an overhead shot of a lined lot.")
        st.stop()

    st.markdown(
        """
        <div class="legend">
          <span><i class="swatch" style="background:#22c55e"></i>Empty</span>
          <span><i class="swatch" style="background:#ef4444"></i>Occupied</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    plate = reference()
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    reading = read_lot(bgr, plate, occupied_below=occupied_below)
    if not reading.slots:
        with st.spinner("Finding bays in this overhead shot..."):
            reading = read_overhead(bgr)

    if not reading.slots:
        st.image(image, width="stretch")
        st.warning(
            "No lined bays were found in this photo. Use an overhead shot of a parking lot "
            "where the bays are visible, or pick a sample frame."
        )
        st.stop()

    shown = cv2.cvtColor(reading.aligned_bgr, cv2.COLOR_BGR2RGB)
    occupied, empty = count_slots(reading.slots)
    insights = summarize(occupied, empty)
    annotated = annotate(shown, reading.slots)

    left, right = st.columns(2)
    with left:
        st.subheader("Camera frame")
        st.image(shown, width="stretch")
    with right:
        st.subheader("Slot overlay")
        st.image(annotated, width="stretch")

    total, occ_col, available, occ = st.columns(4)
    total.metric("Total slots", insights.total)
    occ_col.metric("Occupied", insights.occupied)
    available.metric("Available", insights.available)
    occ.metric("Occupancy", f"{insights.occupancy_pct:.1f}%")

    css = level_class(insights.level)
    st.markdown(
        f"""
        <div class="banner {css}">
          <h3>Congestion: {insights.level}</h3>
          <p>{insights.recommendation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if reading.same_camera:
        st.caption(
            "This frame matched the sample camera. Bay positions come from that view "
            "(Almeida et al., 2015, CC BY 4.0). The empty plate is samples/lot-11.jpg."
        )
    elif reading.source == "lines":
        st.caption(
            "These bays were traced from the painted lines in this overhead shot. "
            "Empty asphalt is green. A bay with a vehicle in it is red."
        )
    else:
        st.caption(
            "These bays were found in the overhead photo itself, so the shot does not "
            "have to be the sample camera. The reader is a YOLOv8n model trained on PKLot "
            "(vacant and occupied spaces)."
        )


if __name__ == "__main__":
    main()

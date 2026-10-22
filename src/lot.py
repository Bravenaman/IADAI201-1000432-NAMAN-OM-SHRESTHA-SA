"""Slot occupancy for the fixed parking-lot camera.

The sample frames are the same viewpoint, already stretched to 640×640.
Painted bays line up on a 5% horizontal grid. Each bay is compared with an
empty plate of this lot. Vehicles left on that plate are filled from a
neighboring empty bay first. A bay is occupied when it no longer matches
the plate, or when it is only a partial match but much busier than the
empty asphalt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.detect import Slot

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

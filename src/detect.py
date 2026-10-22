"""Draw occupied and empty parking slots on a lot image."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

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

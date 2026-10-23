"""ParkVision AI dashboard.

Upload an overhead shot of a lined parking lot. Bays that match the sample
camera use that camera's map. Any other overhead shot is read by a
parking-space model: green for an empty bay, red for an occupied one.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.detect import annotate, count_slots
from src.insights import summarize
from src.lot import OCCUPIED_BELOW, load_reference, read_lot
from src.overhead import read_overhead

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"

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


@st.cache_resource(show_spinner=False)
def reference():
    return load_reference(SAMPLES)


def list_samples() -> list[Path]:
    if not SAMPLES.is_dir():
        return []
    return sorted(SAMPLES.glob("*.jpg")) + sorted(SAMPLES.glob("*.png"))


def read_rgb(file) -> np.ndarray:
    return np.array(Image.open(file).convert("RGB"))


def level_class(level: str) -> str:
    return {"Low": "low", "Moderate": "moderate", "High": "high"}.get(level, "none")


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

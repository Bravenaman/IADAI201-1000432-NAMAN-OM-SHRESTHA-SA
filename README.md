# IADAI201-1000432-NAMAN-OM-SHRESTHA-SA
# ParkVision AI — Overhead Parking Lot Occupancy Detection

An end-to-end computer vision application built with **OpenCV**, **Ultralytics YOLOv8n**, and **Streamlit** to read an overhead parking lot, mark every bay as occupied or empty, and tell a driver whether the lot is open, busy, or full.

---

## 📌 Project Overview

Finding a bay from a fixed camera, or from a single overhead photo of a different lot, is the job of this page. ParkVision AI is a lightweight, deployable lot desk. It does not ask the user to draw boxes. It reads the frame, paints each stall, and turns the counts into a congestion call.

The saved frames are one camera from the PKLot dataset, already resized to 640×640. That camera has a known empty plate, so each bay is compared with the same bay on the plate. An upload of a different lot is not forced onto that map. The page then uses one of three readers:

1. **Sample camera** (the fixed grid against the empty plate `samples/lot-11.jpg`)
2. **Painted lines** (straight stalls, diagonal rows, and herringbone stalls boxed along the slant)
3. **PKLot model** (YOLOv8n weights in `models/pklot-yolov8n.pt`, classes `vacant` and `occupied`)

The lot-level call is not a fixed percentage such as “80% means full.” The seventeen saved frames are averaged first. On this camera that average is about **46 open bays out of 144**. Twice that share of open bays is free. Half of it is full. The same fractions are scaled to the lot in the photo, so a 36-bay herringbone lot and the 144-bay sample camera are judged on the same rule.

A frame with no visible lined bays is reported as **No slots detected**. It is not treated as an empty lot, and it does not play a sound.

The images are CC BY 4.0, Universidade Federal do Paraná. Cite them as:

> Almeida, P., Oliveira, L. S., Silva Jr, E., Britto Jr, A., and Koerich, A. PKLot – A robust dataset for parking lot classification. *Expert Systems with Applications*, 42(11):4937–4949, 2015.

`docs/README.dataset.txt` and `docs/README.roboflow.txt` describe the 12,416-image, 640×640 export the weights were trained on.

---

## 🚀 Key Features

- **Three readers, chosen from the photo:**
  - **Sample-camera grid:** Lines the shot up to `samples/lot-11.jpg` and scores every calibrated bay.
  - **Painted-line geometry:** Follows the white stall dividers. Straight lots stay on upright boxes. Diagonal and herringbone lots are boxed along the slant, one stall at a time, not across the driving lane.
  - **PKLot YOLOv8n:** Used when the line reader cannot see enough bays. The model runs at confidence 0.30 and image size 960, and a sideways photo is tried upright, upside down, and on both sides.
- **Open stalls that have no car beside them:** A gap between two cars on the same row is marked empty. A whole empty column, which has no car to start from, is marked empty only when both painted sides run the length of the stall and the middle is bare asphalt.
- **Lot desk UI:**
  - 🖼️ **Split / Overlay / Camera:** The raw frame and the boxed frame, or either one on its own.
  - 🎛️ **Saved frames:** Chips `01`–`17` for the sample camera. `lot-01` is nearly full. `lot-12` is mostly open.
  - 📤 **Upload:** Any other overhead JPG, PNG, or WEBP. It does not have to be the sample camera.
  - 🟢🔴 **Bay squares:** One cell per stall, green when empty and red when occupied, ordered from the top of the frame.
- **Congestion call and sound:** Low, Moderate, or High, with the recommendation on the page. A short chime, a warning ding, or a 3 second alarm starts on its own, including when the next frame falls in the same band. There is no play button.
- **Registration:** A larger file, a small tilt, or a crop of the sample camera is aligned to the empty plate before the grid is scored, so the boxes stay on the stalls instead of the road.
- **Safe empty state:** If no lined bays are found, the page says so. It does not call that photo a free lot.

---

## 🛠️ System Architecture & Workflow

The whole pipeline is `app.py`. The browser only displays the result Streamlit sends back.

```text
[Input: saved frame 01–17  /  uploaded overhead photo]
                    │
                    ▼
          [Decode to RGB, then BGR]
                    │
                    ▼
        [Does this match the sample camera?]
        align to samples/lot-11.jpg (empty plate)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
 [Yes: fixed bay grid]   [No: other overhead lot]
 compare each bay to      painted stall lines
 the empty plate          ├── straight upright boxes
 (correlation + texture)  ├── diagonal / herringbone
                          │     quads along the slant
                          │     + open stalls between
                          │       painted sides
                          └── too few bays?
                                PKLot YOLOv8n
                                (4 orientations)
                    │
                    ▼
         [Count occupied and empty]
   occupancy %  and  open-bay share
                    │
                    ▼
     [Compare with the saved-frame average]
  ├── open ≥ 2× average ──> Low    ──> short chime
  ├── open ≤ ½× average ──> High   ──> 3 second alarm
  └── in between        ──> Moderate ──> warning ding
                    │
                    ▼
          [Streamlit lot desk]
  ├── Camera frame and colour-coded overlay
  ├── Occupancy ring, totals, bay squares
  └── Recommendation and the band sound
```

**Sample-camera path.** `read_lot` checks whether the photo registers to the empty plate. The grid covers seven lined asphalt rows, the cobblestone row under the grass median, and the row at the bottom edge. The driving aisle between those lower rows is not a bay, and the drive lane on the right is cut off so bare asphalt is not counted as a free stall. Two vehicles that were still parked in `lot-11.jpg` are filled from the neighbouring empty bay, so they are not learned as empty pavement. The comparison is normalised, so a brighter or darker day still matches an empty bay.

**Other-lot path.** `read_overhead` runs the line reader and the PKLot model. The line reader wins when it finds more bays (at least eight). Inside that reader, diagonal paint is separated from upright and flat paint. Herringbone lots have two slant families. Each family is turned so its dividers stand upright, cars are taken as tall blobs, and open stalls are filled along the row. If the lines are too faint, the YOLO boxes are kept instead. A photo that yields nothing from either reader stops with “no slots detected.”

---

## 📂 Project Structure

```text
.
├── app.py                         # Streamlit page and the full reading pipeline
├── requirements.txt               # Python packages (Streamlit, OpenCV, Ultralytics, Pillow)
├── packages.txt                   # System libraries OpenCV needs on Streamlit Cloud
├── assets/logo.png                # Header mark (also embedded in app.py)
├── samples/
│   ├── lot-01.jpg … lot-17.jpg    # Seventeen frames from the sample camera
│   └── lot-11.jpg                 # Empty plate for that camera
├── models/pklot-yolov8n.pt        # YOLOv8n weights (vacant / occupied)
├── notebooks/train_parkvision.ipynb
│                                  # Optional fine-tune on the PKLot 640 export
├── docs/                          # Notes on the PKLot dataset and the Roboflow export
├── tests/                         # Grid, congestion bands, alarm length, slanted lots
│   └── fixtures/                  # Straight, diagonal, herringbone, and top-down photos
├── .streamlit/config.toml         # Local port 8765 and the dark theme
└── README.md                      # This file
```

The page does not train at startup. The notebook is only for replacing the weights later.

---

## ⚙️ Installation & Local Setup

### 1. Prerequisites
- Python 3.10 or newer.
- On Linux, the OpenCV system libraries in `packages.txt` (`libgl1` and `libglib2.0-0`). A normal desktop OpenCV install already has them.

### 2. Clone the Repository
```bash
git clone <your-repo-link>
cd <your-repo-folder>
```

### 3. Create a Virtual Environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies
`requirements.txt` already lists:

```text
streamlit>=1.40
numpy>=1.26
pillow>=10.0
pandas>=2.2
pyyaml>=6.0
ultralytics>=8.3
opencv-python-headless>=4.10
```

Then run:

```bash
pip install -r requirements.txt
```

If the app stops on `import cv2`, the system libraries from `packages.txt` are missing. On Debian or Ubuntu:

```bash
sudo apt-get install -y libgl1 libglib2.0-0
```

### 5. Keep the data files next to `app.py`
- `samples/lot-01.jpg` through `samples/lot-17.jpg`, including the empty plate `samples/lot-11.jpg`.
- `models/pklot-yolov8n.pt`. Git ignores other `*.pt` files, but this weights file is the one the page loads. The weights are AGPL-3.0, from [dronefreak/pklot-yolov8n](https://huggingface.co/dronefreak/pklot-yolov8n).
- `assets/logo.png` is optional. The same mark is stored inside `app.py`, so a checkout without the `assets/` folder still opens.

### 6. Run the Application
```bash
streamlit run app.py
```

`.streamlit/config.toml` sets the port to **8765**. Open `http://localhost:8765`. The page starts on sample frame `01`.

Check the reading logic without the browser:

```bash
python -m pytest tests -q
```

---

## 📊 Evaluation & Classification Logic

### Bay status

| Bay state | How it is decided | What the page draws |
| :--- | :--- | :--- |
| **Occupied** (sample camera) | Normalised match with the empty plate below 0.55 (0.50 on the cobblestone row). A bay that still roughly matches, but is much busier than the empty asphalt (texture ratio above 4.5), is occupied too. On the bottom row, tree shadow alone is not enough: a mild mismatch counts only when the bay is also much busier than bare cobblestone. | Red box `(239, 68, 68)` |
| **Empty** (sample camera) | The bay still matches the empty plate and is not much busier than bare pavement. | Green box `(34, 197, 94)` |
| **Occupied** (painted lines) | A vehicle-shaped blob whose long axis follows the stall, or a gap between two detected cars whose core is clearly a vehicle rather than flat asphalt. | Red box, upright on a straight lot and slanted on a diagonal or herringbone lot |
| **Empty** (painted lines) | Flat asphalt one stall step from a car, still in line with the next car on that row, or a stall whose two painted sides run its length and whose middle stays close to the asphalt tone. Entrance ticks in the driving lane are rejected. | Green box on that one stall |
| **Model bay** | YOLOv8n class `occupied` or `vacant`, confidence 0.30, image size 960. Kept only when the line reader finds fewer bays. | Red or green upright box |
| **No slots detected** | Neither the grid, the lines, nor the model found a bay. | The photo, plus a warning. No sound. |

### Lot call

The average open share is measured on the seventeen saved frames (about 46 open bays, roughly 32% of 144). For a lot with `T` bays, full is at or under half of that share, and free is at or over twice that share.

| Open bays, compared with the average | Congestion | Sound | Recommendation |
| :--- | :--- | :--- | :--- |
| At least twice the average | **Low** | Short chime | Slots available — proceed. |
| Between half and twice the average | **Moderate** | Warning ding | Some slots are open — proceed, and take the nearest empty bay. |
| Half the average or fewer | **High** | 3 second alarm | Parking full — try another area. |

The sound starts by itself when the reading finishes, and it starts again if the next frame is in the same band. `samples/lot-01.jpg` should land toward High or Moderate. `samples/lot-12.jpg` should land on Low, with far more green boxes than red ones.

---

## ☁️ Deployment on Streamlit Cloud

1. Push this folder to **GitHub**. The deploy must include `app.py`, `requirements.txt`, `packages.txt`, `samples/`, and `models/pklot-yolov8n.pt`.
2. Go to [share.streamlit.io](https://iadai201-1000432-naman-om-shrestha-sa-l6hfmpa9wtdp8myatyrmu8.streamlit.app/) and sign in with GitHub.
3. Select this repository, branch `main`, and set the main file path to `app.py`.
4. Deploy. The first boot installs the Python packages and the system libraries in `packages.txt`. Without `packages.txt` the log stops on `import cv2`.
5. If `packages.txt` is added or edited after the first deploy, open **Manage app** and reboot. A normal code push does not reinstall system libraries.
6. Open the app URL Streamlit prints. The page should load frame `01` without a `FileNotFoundError`. The header logo is embedded in `app.py`, so a missing `assets/logo.png` no longer stops startup.

---

## 🔮 Future Improvements & Maintenance

- **Worn stall paint:** An empty herringbone column is marked only when both painted sides are visible for the length of the stall. Broken or occluded lines still leave some open bays unmarked.
- **Cars that match the asphalt:** A grey or black roof with almost no contrast can be missed by the blob detector. The line reader will not invent a box unless the stall sides are clear.
- **A baseline per lot:** Full and free are scaled from the sample camera’s average. A lot whose usual occupancy is very different would want its own saved frames and its own average.
- **Live camera or a short clip:** The page reads one still photo. A fixed CCTV URL, or a few seconds of video with the same band logic, is not wired in.
- **Retraining:** `notebooks/train_parkvision.ipynb` downloads the public PKLot v2 640 export, resplits it 70 / 15 / 15, and fine-tunes YOLOv8n. The running app does not train. Replacing `models/pklot-yolov8n.pt` is a manual step, and that file has to stay tracked in git.
- **Do not replace the empty plate:** `samples/lot-11.jpg` is what every sample-camera count is compared with. Swapping it changes the whole grid.
- **After a logic change:** Run `python -m pytest tests -q`. The tests cover the sample grid, the three congestion bands, the 3 second alarm, and the straight, diagonal, and herringbone fixtures.
- **On Streamlit Cloud:** Reboot from **Manage app** after any edit to `packages.txt`, and confirm `import cv2` succeeds before judging a photo.

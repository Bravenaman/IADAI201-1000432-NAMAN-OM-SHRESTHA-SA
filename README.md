# ParkVision AI

ParkVision reads an overhead shot of a lined parking lot and marks each bay as occupied or empty. It then reports how full the lot is and tells a driver whether to pull in or try another area.

The running app is Streamlit. There is no training step to launch it.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints. The default port in `.streamlit/config.toml` is 8765. The sidebar starts on a sample frame from `samples/`.

## What you see

- The camera frame and the same frame with a bay overlay
- Green boxes for empty bays, red boxes for occupied bays
- Total bays, occupied bays, available bays, and occupancy
- A congestion level and a recommendation

| Occupancy | Congestion | Recommendation |
| --- | --- | --- |
| under 40% | Low | Slots available — proceed. |
| 40% to 75% | Moderate | Some slots are open — proceed, and take the nearest empty bay. |
| over 75% | High | Parking full — try another area. |

## How a bay is judged

These frames are one fixed camera, already resized to 640×640 (the PKLot / Roboflow 640 export). The painted bays in the asphalt rows fall on a regular grid, so the app does not need a separate detection model to find them.

`samples/lot-11.jpg` is the empty plate: a cloudy frame with the lot almost vacant. Two vehicles still parked in that frame are filled in from the neighboring empty bay, so they are not learned as empty asphalt. Each bay in a new frame is compared with the same bay on that plate. A poor match means a vehicle is there. A bay that only partly matches, but is much busier than the empty asphalt, is counted as occupied too. The comparison is normalised, so a brighter or darker day still matches an empty bay.

The grid covers the seven lined asphalt rows. The driving aisles are not bays. The tree-covered lower court is outside the grid, so cars under those trees are not counted.

The sample frames are one camera. An upload of that same view is lined up to it before the bays are read, so a larger file, a small tilt, or a crop that still shows the lined rows lands on the stalls instead of the road.

An overhead shot of a different lot is not forced onto that map. A surveillance-style lot uses `models/pklot-yolov8n.pt`, a YOLOv8n detector trained on PKLot. Those weights are AGPL-3.0, published by [dronefreak/pklot-yolov8n](https://huggingface.co/dronefreak/pklot-yolov8n). Classes are `vacant` and `occupied`. A straight top-down photo, with the bays painted on the asphalt, is read from those lines instead: empty asphalt is a free bay, and a vehicle in the bay is occupied. A sideways photo is turned until the bays are found. If no lined bays are visible, the page says so and does not call the lot full.

## Samples

`samples/lot-01.jpg` through `samples/lot-17.jpg` are frames from this viewpoint at different times of day and different occupancy. `lot-01` is nearly full. `lot-12` is mostly open.

## Reference

Almeida, P., Oliveira, L. S., Silva Jr, E., Britto Jr, A., and Koerich, A. PKLot – A robust dataset for parking lot classification. *Expert Systems with Applications*, 42(11):4937–4949, 2015.

The images are CC BY 4.0, Universidade Federal do Paraná. `docs/README.dataset.txt` and `docs/README.roboflow.txt` describe the 640×640 export.

# Smart Composter Classification

A YOLOv8 model that classifies waste in a live webcam feed for a smart composter. Items detected in a central region of interest (ROI) are checked: if any `Non_Organik` (non-organic) item is present, the item is rejected; otherwise the organic types are shown.

## Files

| File | Purpose |
|------|---------|
| `Model_Training_Code.ipynb` | Downloads the dataset from Roboflow, trains YOLOv8n (50 epochs, 640px), validates, and exports `best_v2.pt`. Written for Kaggle/Colab with a GPU. |
| `Test_Model_Code.py` | Runs the trained model on a webcam feed with the ROI overlay and accept/reject status. Press `q` to quit. |

## The model file (`best_v2.pt`) is not included

Trained weights are not stored in this repository (`*.pt` is in `.gitignore`). You need to produce or obtain `best_v2.pt` yourself:

1. Run `Model_Training_Code.ipynb` (see below). The last cell downloads the trained weights to your browser as `best_v2.pt`. (It reads them from `/kaggle/working/runs/detect/train/weights/best.pt`; adjust that path if you train elsewhere.)
2. Place `best_v2.pt` in the same folder as `Test_Model_Code.py`.

## Setup

Requires Python 3.9+ and a webcam.

```bash
pip install ultralytics roboflow opencv-python
```

### Training

1. Create a Roboflow API key in your Roboflow account settings.
2. Set these environment variables. The notebook reads them and never contains the values itself (`ROBOFLOW_WORKSPACE` and `ROBOFLOW_PROJECT` are the workspace and project slugs of your Roboflow dataset, version 2 is used):
   - Linux/macOS: `export ROBOFLOW_API_KEY="your_key"`
   - Windows PowerShell: `$env:ROBOFLOW_API_KEY = "your_key"`
   - Kaggle: add it under Add-ons > Secrets and expose it with `os.environ["ROBOFLOW_API_KEY"] = ...`
3. Run the notebook cells in order.

Set `ROBOFLOW_API_KEY`, `ROBOFLOW_WORKSPACE` and `ROBOFLOW_PROJECT` the same way. The author's dataset is private, so point them at your own Roboflow dataset (YOLOv8 format). The notebook has no saved outputs; run the cells to see training logs and the mAP scores.

### Testing with a webcam

```bash
python Test_Model_Code.py
```

Tunable constants at the top of the script:

- `DETECT_INTERVAL` (0.2 s): time between inference runs
- `CONF_THRESHOLD` (0.5): minimum detection confidence
- `OVERLAP_THRESHOLD` (0.5): minimum fraction of a box that must lie inside the ROI

If the wrong camera opens, change `cv2.VideoCapture(0)` to another index.

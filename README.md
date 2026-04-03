# Staff Identifier

A computer vision pipeline designed to identify and localize staff members in overhead retail footage by detecting a badge.

 1. Overview
The goal of this project is to scan video from overhead wide-angle cameras to identify frames containing staff members. A staff member is defined as an individual wearing a visible name tag or badge. For every detection, the system reports the precise (X, Y) pixel coordinates of the person.

 **Input:** 960×720px, 25fps fisheye video (approx. 53 seconds/1341 frames).
 **Output:** `results.csv` and an `annotated_frames/` directory for visual verification.

## 2. Technical Approach
The solution implements a two-step **cascade architecture** to balance processing speed with detection accuracy.

### Step 1: Person Detection (YOLOv8)
The pipeline uses **YOLOv8-nano** to detect all people (COCO class 0) in each sampled frame. This model works out-of-the-box for overhead angles without custom training. 
* **Fallback:** If the YOLO environment is unavailable, the script automatically switches to OpenCV’s built-in **HOG (Histogram of Oriented Gradients)** person detector via the `--fallback` flag.

### Step 2: Name Tag Detection (OpenCV)
Each detected person is cropped and analyzed for a badge using classical computer vision heuristics:
* **Region of Interest:** Crops the top 60% of the bounding box to focus on the chest area.
* **Color Filtering:** Converts to **HSV color-space** and thresholds for low-saturation, high-brightness pixels (targeting white/light-colored badges).
* **Geometric Validation:** Contours are filtered by size (1–25% of crop area), solidity (>0.6 to ensure rectangularity), and aspect ratio (0.5–4.0).
* **Localization:** The reported XY coordinate is the centroid of the staff member's bounding box (Origin: top-left).

## 3. Pipeline Summary

| Step | Process | Technology | Output |
| :--- | :--- | :--- | :--- |
| 1 | Frame Sampling | OpenCV VideoCapture | Raw frames |
| 2 | Person Detection | YOLOv8-nano (Conf ≥ 0.4) | Bounding boxes |
| 3 | Badge Check | HSV Threshold + Contour Filter | Staff flag (0/1) + XY |
| 4 | Save Results | Python CSV + OpenCV | results.csv + JPEGs |



## 4. How to Run

### Installation
```bash
pip install ultralytics opencv-python
```

### Usage
```bash
# Basic run
python staff_detection.py sample.mp4

# Denser sampling (every 2nd frame)
python staff_detection.py sample.mp4 --sample-rate 2

# Custom output folder
python staff_detection.py sample.mp4 --output results/

# Use HOG fallback
python staff_detection.py sample.mp4 --fallback
```

## 5. Outputs
* **`output/results.csv`**: Contains frame index, timestamp (s), staff flag, and centroid coordinates (`cx`, `cy`).
* **`output/annotated_frames/`**: Contains a JPEG for every positive detection, labeled with a green bounding box and centroid dot.

## 6. Assumptions & Limitations
* **Single Staff Assumption:** The current logic logs the first staff member detected per frame.
* **Visibility:** Detection requires the badge to be visible; staff with their backs turned will be missed.
* **Color Sensitivity:** The HSV threshold (Value > 170) is tuned for white/light badges. Darker badges would require alternative thresholding or a trained classifier.
* **Future Roadmap:** To improve robustness against lighting shifts, Stage 2 could be replaced by a fine-tuned lightweight classifier like **MobileNet**.

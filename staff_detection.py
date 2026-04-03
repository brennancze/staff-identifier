import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Try loading YOLOv8, fall back to HOG if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# Person detection 

def load_detector(use_hog=False):
    if not use_hog and YOLO_AVAILABLE:
        print("[INFO] Loading YOLOv8-nano...")
        model = YOLO("yolov8n.pt")
        return ("yolo", model)
    else:
        print("[INFO] Using HOG person detector (fallback)...")
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        return ("hog", hog)


def detect_people(detector, frame):
    """
    Returns list of (x1, y1, x2, y2) bounding boxes for detected people.
    """
    dtype, model = detector
    boxes = []

    if dtype == "yolo":
        results = model(frame, classes=[0], conf=0.4, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes.append((x1, y1, x2, y2))

    elif dtype == "hog":
        # Resize for HOG speed
        scale = 0.5
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rects, _ = model.detectMultiScale(small, winStride=(8, 8), padding=(4, 4), scale=1.05)
        for (x, y, w, h) in rects:
            boxes.append((
                int(x / scale), int(y / scale),
                int((x + w) / scale), int((y + h) / scale)
            ))

    return boxes


# Name tag detection 

def has_name_tag(crop):
   
    h, w = crop.shape[:2]
    if h < 30 or w < 20:
        return False

    # Focus on upper-body (top 60% of crop) where badge would be
    upper = crop[:int(h * 0.65), :]

    # Convert to HSV
    hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)

    # Threshold for white/bright badge region
    badge_mask = cv2.inRange(hsv, (0, 0, 170), (180, 55, 255))

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    badge_mask = cv2.morphologyEx(badge_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    badge_mask = cv2.dilate(badge_mask, kernel, iterations=2)

    # Find contours in the bright region
    contours, _ = cv2.findContours(badge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Badge should be a reasonably sized but not huge region
        crop_area = upper.shape[0] * upper.shape[1]
        rel_area = area / crop_area

        if rel_area < 0.01 or rel_area > 0.30:
            continue

        # Check it's roughly rectangular (high solidity)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < 0.6:
            continue

        # Check aspect ratio of bounding rectangle 
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh == 0:
            continue
        aspect = bw / bh
        if 0.5 < aspect < 4.0:
            return True

    return False


# Main pipeline

def process_video(video_path, sample_rate=3, output_dir="output", use_hog=False):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    frames_out = Path(output_dir) / "annotated_frames"
    frames_out.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video   : {video_path}")
    print(f"[INFO] Frames  : {total}  |  FPS: {fps:.1f}  |  Size: {width}x{height}")
    print(f"[INFO] Sampling every {sample_rate} frame(s)\n")

    detector = load_detector(use_hog)

    csv_rows   = []
    staff_hits = []
    frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            ts = frame_idx / fps
            print(f"  Processing frame {frame_idx:5d}/{total}  ({ts:.1f}s)...", end="\r")

            people = detect_people(detector, frame)
            found_staff = False
            staff_cx, staff_cy = None, None
            staff_bbox = None

            for (x1, y1, x2, y2) in people:
                # Clamp to frame bounds
                x1c = max(0, x1); y1c = max(0, y1)
                x2c = min(width, x2); y2c = min(height, y2)
                crop = frame[y1c:y2c, x1c:x2c]
                if crop.size == 0:
                    continue

                if has_name_tag(crop):
                    found_staff = True
                    staff_cx    = (x1c + x2c) // 2
                    staff_cy    = (y1c + y2c) // 2
                    staff_bbox  = (x1c, y1c, x2c, y2c)
                    break  # one staff at a time

            if found_staff and staff_bbox:
                # Save annotated frame
                vis = frame.copy()
                x1, y1, x2, y2 = staff_bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 60), 3)
                cv2.circle(vis, (staff_cx, staff_cy), 8, (0, 220, 60), -1)
                label = f"STAFF  f:{frame_idx}  ({staff_cx},{staff_cy})"
                cv2.putText(vis, label, (x1, max(y1 - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 60), 2)
                cv2.imwrite(str(frames_out / f"frame_{frame_idx:05d}.jpg"), vis)
                staff_hits.append((frame_idx, ts, staff_cx, staff_cy))

            csv_rows.append({
                "frame":       frame_idx,
                "timestamp_s": round(ts, 3),
                "staff":       int(found_staff),
                "cx":          staff_cx if found_staff else "",
                "cy":          staff_cy if found_staff else "",
            })

        frame_idx += 1

    cap.release()

    # Save CSV
    csv_path = Path(output_dir) / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "timestamp_s", "staff", "cx", "cy"])
        w.writeheader()
        w.writerows(csv_rows)

    # Print summary
    print(f"\n\n{'─'*55}")
    print(f"  Done. Processed {len(csv_rows)} sampled frames.")
    print(f"  Staff detected in {len(staff_hits)} frame(s).")
    print(f"  Results saved to : {csv_path}")
    print(f"  Annotated frames : {frames_out}/")
    print(f"{'─'*55}\n")

    if staff_hits:
        print(f"  {'Frame':>6}  {'Time (s)':>8}  {'CX':>5}  {'CY':>5}")
        print(f"  {'─'*38}")
        for f, t, cx, cy in staff_hits:
            print(f"  {f:>6}  {t:>8.2f}  {cx:>5}  {cy:>5}")
    else:
        print("  No staff detected. Try --sample-rate 1 or check thresholds.")

    return staff_hits


# CLI 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Staff name-tag detector")
    parser.add_argument("video",                         help="Input video path")
    parser.add_argument("--sample-rate", type=int, default=3,
                        help="Process every N frames (default: 3)")
    parser.add_argument("--output",      default="output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--fallback",    action="store_true",
                        help="Use HOG detector instead of YOLO")
    args = parser.parse_args()

    if not Path(args.video).exists():
        sys.exit(f"[ERROR] File not found: {args.video}")

    process_video(
        video_path  = args.video,
        sample_rate = args.sample_rate,
        output_dir  = args.output,
        use_hog     = args.fallback,
    )

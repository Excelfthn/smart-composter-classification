import cv2
import time
from ultralytics import YOLO

model = YOLO("best_v2.pt")
cap = cv2.VideoCapture(0)

DETECT_INTERVAL = 0.2
CONF_THRESHOLD = 0.5
OVERLAP_THRESHOLD = 0.5

last_detect_time = 0
detected_classes = set()
last_boxes = []


def hitung_overlap_ratio(box, roi):
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = roi

    ix1, iy1 = max(x1, rx1), max(y1, ry1)
    ix2, iy2 = min(x2, rx2), min(y2, ry2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    area_overlap = (ix2 - ix1) * (iy2 - iy1)
    area_box = (x2 - x1) * (y2 - y1)
    return area_overlap / area_box


while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    roi_w, roi_h = int(w * 0.4), int(h * 0.4)
    roi_x1, roi_y1 = (w - roi_w) // 2, (h - roi_h) // 2
    roi_x2, roi_y2 = roi_x1 + roi_w, roi_y1 + roi_h

    now = time.time()
    if now - last_detect_time >= DETECT_INTERVAL:
        last_detect_time = now
        t_start = time.time()
        results = model(frame, verbose=False)[0]
        print(f"Waktu inference: {time.time() - t_start:.2f} detik")
        detected_classes = set()
        last_boxes = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            overlap = hitung_overlap_ratio((x1, y1, x2, y2), (roi_x1, roi_y1, roi_x2, roi_y2))

            if overlap >= OVERLAP_THRESHOLD:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                detected_classes.add(cls_name)
                last_boxes.append((x1, y1, x2, y2, cls_name, conf))

    for x1, y1, x2, y2, cls_name, conf in last_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)

    if detected_classes:
        if "Non_Organik" in detected_classes:
            status = "DITOLAK (ada non-organik)"
            color = (0, 0, 220)
        else:
            jenis = ", ".join(sorted(detected_classes))
            status = f"ORGANIK: {jenis}"
            color = (0, 180, 0)

        (tw, th), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (20, 20), (50 + tw, 60 + th), color, -1)
        cv2.putText(frame, status, (30, 50 + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Smart Composter Classifier v2", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
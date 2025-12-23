import cv2
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# ===============================
# CONFIGURATION
# ===============================
MODEL_PATH = "yolov8m_multiscale_face.pt"

CONF_THRESH = 0.4
IOU_THRESH = 0.5

NO_FACE_TIME = 5          # seconds
MULTI_FACE_LIMIT = 1      # more than 1 face = violation
MIN_FACE_AREA = 3000      # camera too far
MAX_FACE_AREA = 90000     # camera too close
MOVE_THRESHOLD = 60       # pixels
DARK_FRAME_THRESH = 30    # camera tamper

# ===============================
# LOAD MODEL
# ===============================
model = YOLO(MODEL_PATH)

# ===============================
# VIDEO SETUP
# ===============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

prev_time = 0
last_face_time = time.time()
prev_center = None
recording = False
video_writer = None

print("""
Controls:
 q  → quit
 s  → save snapshot
 r  → start/stop recording
""")

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)

    results = model.predict(
        source=frame,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        verbose=False
    )

    boxes = results[0].boxes
    annotated = results[0].plot()

    face_count = 0
    face_centers = []

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            area = (x2 - x1) * (y2 - y1)

            if area >= MIN_FACE_AREA:
                face_count += 1
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                face_centers.append((cx, cy))

    # ===============================
    # LOGIC CHECKS
    # ===============================

    alerts = []

    # 1️⃣ No face detection
    if face_count == 0:
        if time.time() - last_face_time > NO_FACE_TIME:
            alerts.append("NO FACE DETECTED")
    else:
        last_face_time = time.time()

    # 2️⃣ Multiple faces
    if face_count > MULTI_FACE_LIMIT:
        alerts.append("MULTIPLE FACES")

    # 3️⃣ Face movement
    if prev_center and face_centers:
        dist = np.linalg.norm(
            np.array(face_centers[0]) - np.array(prev_center)
        )
        if dist > MOVE_THRESHOLD:
            alerts.append("EXCESSIVE MOVEMENT")

    if face_centers:
        prev_center = face_centers[0]

    # 4️⃣ Face size check
    if boxes is not None and face_count > 0:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            area = (x2 - x1) * (y2 - y1)
            if area < MIN_FACE_AREA:
                alerts.append("FACE TOO FAR")
            elif area > MAX_FACE_AREA:
                alerts.append("FACE TOO CLOSE")

    # 5️⃣ Camera tampering
    if avg_brightness < DARK_FRAME_THRESH:
        alerts.append("CAMERA BLOCKED / DARK")

    # ===============================
    # OVERLAYS
    # ===============================
    cv2.putText(annotated, f"Faces: {face_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(annotated, timestamp,
                (20, annotated.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # FPS
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if prev_time else 0
    prev_time = curr_time
    cv2.putText(annotated, f"FPS: {fps}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # ALERT DISPLAY
    for i, alert in enumerate(alerts):
        cv2.putText(annotated, f"⚠ {alert}",
                    (annotated.shape[1] - 350, 40 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    # ===============================
    # RECORDING
    # ===============================
    if recording:
        video_writer.write(annotated)
        cv2.putText(annotated, "● REC",
                    (annotated.shape[1] - 120, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Face Monitoring System", annotated)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('s'):
        name = f"snapshot_{datetime.now().strftime('%H%M%S')}.jpg"
        cv2.imwrite(name, annotated)
        print(f"📸 Snapshot saved: {name}")

    elif key == ord('r'):
        if not recording:
            h, w, _ = frame.shape
            video_writer = cv2.VideoWriter(
                f"recording_{datetime.now().strftime('%H%M%S')}.mp4",
                cv2.VideoWriter_fourcc(*'mp4v'),
                20,
                (w, h)
            )
            recording = True
            print("🔴 Recording started")
        else:
            recording = False
            video_writer.release()
            print("⏹ Recording stopped")

# ===============================
# CLEANUP
# ===============================
cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()

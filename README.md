# 🚀 Multi-Scale Face Detection & Real-Time Face Monitoring System (YOLOv8)

A **production-grade, real-time multi-scale face detection and monitoring system** built using **YOLOv8**, designed to operate reliably in **crowded, unconstrained, real-world environments**.

This project goes far beyond basic face detection by integrating **rule-based analytics**, **behavioral monitoring**, and **camera integrity checks**, making it suitable for **exam proctoring, surveillance analytics, attendance systems, and driver monitoring**.

---

## 🔥 Key Highlights

- ✅ Multi-scale face detection (small, medium, large faces)
- ⚡ Real-time performance using YOLOv8m
- 👥 Person / face counting
- 🚨 Multiple-face & absence violation detection
- 🎥 Camera tampering & dark-frame detection
- 🧍 Face movement & behavior analysis
- 📸 Snapshot capture & video recording
- 📊 FPS monitoring & timestamp logging
- 💻 Runs on Free Colab GPU and local machines

---

## 🧠 Why This Project Matters

Most face-based AI systems fail in real-world scenarios due to:
- Scale variation (near vs far faces)
- Crowded scenes
- Occlusion
- Camera quality issues

This project solves the **FOUNDATIONAL PROBLEM**:

> **Reliable face localization under real-world constraints**

All higher-level systems (emotion detection, drowsiness detection, malpractice detection, face recognition) depend on this step.

---

## 🏗️ System Architecture

```text
Camera / Video Feed
        ↓
YOLOv8m Multi-Scale Face Detector
        ↓
Bounding Boxes (Face Localization)
        ↓
Rule-Based Analytics Layer
        ├── Face Counting
        ├── Absence Detection
        ├── Multi-Face Violation
        ├── Face Size Validation
        ├── Movement Detection
        ├── Camera Tampering Detection
        ↓
Alerts / Logging / Monitoring Output
```
## 🧪 Model & Training Details
- Component	Details
- Model	YOLOv8m (Ultralytics)
- Task	Single-class face detection
- Dataset	Roboflow Face Detection Dataset (~6K images)
- Training Platform	Google Colab (Tesla T4)
- Image Size	640 × 640
- Optimizer	AdamW
- Epochs	Early-stopped at ~20
- Best Accuracy	96.5% mAP@50
- Inference Speed	Real-time (Webcam)

## 📊 Performance Metrics

- mAP@50: 96.5%
- Precision: ~95%

- Recall: ~90%

   mAP@50–95: ~70%

Training was stopped early once validation performance plateaued to avoid overfitting and unnecessary computation.

## 🧩 Implemented Features (No Retraining)
## 👤 Presence & Integrity Monitoring

- Face counting (persons present)

- No-face detection (absence alert)

- Multiple-face violation detection

- Face persistence monitoring

## 🎥 Camera & Quality Checks

- Face size validation (camera too far / too close)

- Camera tampering / dark-frame detection
- 🧍 Behavioral Analysis

- Face movement tracking

- Sudden motion alerts

## 🧠 System Utilities

- FPS calculation

- Timestamp overlay

- Snapshot capture (s)

- Video recording (r)

- Clean exit handling (q)

## 🧑‍💻 Real-Time Demo (Local Machine)
🔧 Installation
```
pip install ultralytics opencv-python numpy
```
## ▶ Run Real-Time Monitoring
- python realtime_face_monitoring.py

## 🎮 Controls
- Key	Action
```
q	Quit application
s	Save snapshot
r	Start/Stop video recording
```
## 📁 Project Structure
```
multiface/
├── realtime_face_monitoring.py
├── realtime_face_detect.py
├── README.md
├── requirements.txt
└── yolov8m_multiscale_face.pt   (not included)
```
## 📦 Model Weights

# The trained model file is NOT included in this repository due to size constraints and best practices.

## 🔽 Download Model
# yolov8m_multiscale_face.pt


# Place the model in the project root before running inference.

## Real-World Applications
- 🎓 Exam Proctoring Systems

- 🚗 Driver Monitoring & Drowsiness Detection

- 🎥 CCTV & Surveillance Analytics

- 🏫 Attendance & Classroom Monitoring

- 🛍️ Retail Footfall Analysis

- 🏙️ Smart City Crowd Monitoring

## 🔮 Future Enhancements
- Face tracking (BoT-SORT / ByteTrack)

- Entry–exit line counting

- Streamlit dashboard

- Database logging & analytics

- Integration with drowsiness & emotion models

- Edge deployment (ONNX / TensorRT / Jetson)

## 🏆 Skills Demonstrated

- Computer Vision & Deep Learning

- Object Detection (YOLO, FPN)

- Multi-Scale Feature Learning

- Real-Time Inference Systems

- Dataset Engineering

- ML System Design

- Deployment-Aware ML

## 📜 License

This project is intended for educational and research purposes.

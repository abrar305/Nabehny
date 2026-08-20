#  Nabehny — AI Smart Classroom Assistant

> An AI-powered classroom assistant for automated attendance, face recognition, behavior detection, and classroom analytics.

##  Overview

**Nabehny** is a Computer Vision-based smart classroom system that analyzes classroom images and videos to:

*  Detect and track students
*  Recognize registered students
*  Automate attendance
*  Detect classroom behaviors
*  Generate classroom analytics

---

##  AI Pipeline

```text
Classroom Video
      ↓
YOLOv8n + ByteTrack
      ↓
Person Tracking
      ↓
InsightFace (buffalo_l)
      ↓
Student Recognition
      ↓
Custom YOLO Model (best.pt)
      ↓
Behavior Detection
      ↓
Attendance & Events
      ↓
Database
      ↓
Streamlit Dashboard
```

### Main Models

* **YOLOv8n** → Person detection
* **ByteTrack** → Person tracking
* **InsightFace / buffalo_l** → Face recognition
* **best.pt** → Classroom behavior detection

---

##  Features

*  Student registration & face embeddings
*  Automatic attendance
*  Image & video processing
*  Person detection & tracking
*  Face recognition
*  Behavior/event detection
*  Attendance & behavior analytics
*  CSV reports

---

##  Technologies

`Python` · `OpenCV` · `YOLO` · `ByteTrack` · `InsightFace` · `NumPy` · `Pandas` · `Streamlit` · `Plotly` · `SQLite`

---

##  Project Structure

```text
Nabehny/
├── core/
│   ├── face_engine.py
│   └── video_processor.py
├── gui/
│   ├── app.py
│   ├── add_student.py
│   └── dashboard.py
├── best.pt
├── yolov8n.pt
├── database.py
├── main.py
└── requirements.txt
```

---

##  Installation

```bash
git clone https://github.com/abrar305/Nabehny.git
cd Nabehny
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run main.py
```

---


##  Presentation

[https://canva.link/4dop0gcca0vx658]

---

##  Team

| Member     | LinkedIn | GitHub |
| ---------- | -------- | ------ |
| **[Abrar Ashraf]** | [Link]   | [[Link]](https://github.com/abrar305) |
| **[Name]** | [Link]   | [Link] |
| **[Name]** | [Link]   | [Link] |
| **[Name]** | [Link]   | [Link] |

---

##  Future Improvements

* Real-time camera processing
* More behavior classes
* Multi-camera support
* Real-time notifications
* Improved analytics

---

##  License

Developed for educational and academic purposes.

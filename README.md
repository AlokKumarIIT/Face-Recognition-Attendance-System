# Face Recognition Attendance System

This is a real-time face recognition-based attendance system using **MTCNN** for face detection and **FaceNet (InceptionResnetV1)** for face embedding. A simple **Tkinter GUI** is provided to handle data entry, training, and recognition.

---

## 🔧 Features

- 📸 **Capture Face Images**: Captures face images from webcam with MTCNN.
- 🧠 **Face Embedding & Training**: Extracts 128-D embeddings using pretrained FaceNet (VGGFace2).
- 🧾 **Attendance Logging**: Logs recognized faces with timestamps into a CSV file.
- 📁 **Embeddings Storage**: Stores known face embeddings in JSON format.
- 🖥️ **User Interface**: Easy-to-use Tkinter GUI for interaction.
- 🔁 **Duplicate Entry Prevention**: Prevents logging same person multiple times in short interval.

---

## 🖥️ GUI Options

1. **Name** and **Roll No** Entry Fields
2. **Capture Images** – Captures 50 face samples using webcam.
3. **Train Model** – Trains and saves embeddings from dataset.
4. **Recognize Face** – Starts live recognition and logs attendance.

---

## 📁 Folder Structure

```text
.
├── dataset/                 # Captured images (name_roll_imgcount.jpg)
├── embeddings/
│   └── embeddings.json      # Stored face embeddings
├── attendance.csv           # Attendance log file (auto-generated)
├── face_attendance.py       # Main application script
└── README.md

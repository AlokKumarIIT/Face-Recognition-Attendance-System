import os
import cv2
import numpy as np
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
import torch
from datetime import datetime, date
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import json
import threading
import logging

logging.basicConfig(level=logging.INFO)

root = tk.Tk()
root.title('Face Recognition Attendance System')
root.geometry('400x400')

name_var = tk.StringVar()
roll_var = tk.StringVar()

daily_attendance = set() 
last_logged_time = {} 

detector = MTCNN()

model = InceptionResnetV1(pretrained='vggface2').eval()

known_faces = {}

def preprocess_image(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = torch.tensor(face_img).permute(2, 0, 1).float() / 255.0
    face_img = (face_img - 0.5) / 0.5 
    return face_img.unsqueeze(0)

def get_embedding(face_img):
    with torch.no_grad():
        embedding = model(preprocess_image(face_img))
    return embedding.squeeze().numpy()

def load_embeddings():
    global known_faces
    if os.path.exists('embeddings/embeddings.json'):
        with open('embeddings/embeddings.json', 'r') as f:
            data = json.load(f)
            known_faces = {name: np.array(emb) for name, emb in data.items()}
        logging.info(f"Loaded {len(known_faces)} embeddings from embeddings.json")
    else:
        logging.warning("No embeddings file found. Please train the model.")

def save_embeddings():
    if not os.path.exists('embeddings'):
        os.makedirs('embeddings')
    with open('embeddings/embeddings.json', 'w') as f:
        json.dump({key: value.tolist() for key, value in known_faces.items()}, f)
    logging.info(f"Saved {len(known_faces)} embeddings to embeddings.json")

def capture_images(name, roll):
    cam = cv2.VideoCapture(0)
    count = 0

    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    while True:
        ret, img = cam.read()
        if not ret:
            break
        result = detector.detect_faces(img)

        for face in result:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            face_img = img[y:y+h, x:x+w]

            count += 1
            cv2.imwrite(f'dataset/{name}_{roll}_{count}.jpg', face_img)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Capturing Images', img)

        if cv2.waitKey(100) & 0xFF == ord('q') or count >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()
    messagebox.showinfo('Info', f'Images Saved for {name} with Roll No {roll}')

def train_model():
    global known_faces
    known_faces.clear()

    image_paths = [os.path.join('dataset', f) for f in os.listdir('dataset') if f.endswith('.jpg')]

    for image_path in image_paths:
        img = cv2.imread(image_path)
        result = detector.detect_faces(img)
        if result:
            x, y, w, h = result[0]['box']
            x, y = max(0, x), max(0, y)
            face_img = img[y:y+h, x:x+w]

            embedding = get_embedding(face_img)
            name_roll, _ = os.path.splitext(os.path.basename(image_path))
            name, roll, _ = name_roll.split('_')
            known_faces[f"{name}_{roll}"] = embedding

    save_embeddings()
    messagebox.showinfo('Info', 'Training Completed')

def log_attendance(name, roll, cooldown_seconds=60):
    current_time = datetime.now()
    key = (name, roll)

    if key in last_logged_time:
        time_diff = (current_time - last_logged_time[key]).total_seconds()
        if time_diff < cooldown_seconds:
            logging.info(f"Attendance already logged for {name} recently.")
            return

    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
    log = {'Name': name, 'Roll': roll, 'Timestamp': timestamp}
    df = pd.DataFrame([log])

    if os.path.exists('attendance.csv'):
        df.to_csv('attendance.csv', mode='a', header=False, index=False)
    else:
        df.to_csv('attendance.csv', mode='w', header=True, index=False)

    last_logged_time[key] = current_time
    logging.info(f"Attendance logged for {name} ({roll}) at {timestamp}")

def recognize_face(threshold=0.8):
    if not known_faces:
        messagebox.showwarning('Warning', 'No embeddings found. Please train the model first.')
        return

    cam = cv2.VideoCapture(0)
    current_date = date.today().strftime('%Y-%m-%d')

    while True:
        ret, img = cam.read()
        if not ret:
            break

        result = detector.detect_faces(img)

        for face in result:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)  
            face_img = img[y:y+h, x:x+w]

            embedding = get_embedding(face_img)

            min_dist = float('inf')
            recognized_name = "Unknown"

            for name, known_embedding in known_faces.items():
                dist = np.linalg.norm(embedding - known_embedding)  
                if dist < min_dist:
                    min_dist = dist
                    if dist < threshold: 
                        recognized_name = name

            if recognized_name != "Unknown":
                name, roll = recognized_name.split('_')
                if (name, roll, current_date) not in daily_attendance: 
                    daily_attendance.add((name, roll, current_date))
                    threading.Thread(target=log_attendance, args=(name, roll)).start()

            label = f"{recognized_name} ({min_dist:.2f})"
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Recognizing Face', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def capture_images_click():
    name = name_var.get()
    roll = roll_var.get()
    if name and roll:
        capture_images(name, roll)
    else:
        messagebox.showwarning('Warning', 'Please enter both name and roll number')

def train_model_click():
    train_model()

def recognize_face_click():
    recognize_face(threshold=0.8)

name_label = tk.Label(root, text='Name:')
name_label.pack()
name_entry = tk.Entry(root, textvariable=name_var)
name_entry.pack()

roll_label = tk.Label(root, text='Roll No:')
roll_label.pack()
roll_entry = tk.Entry(root, textvariable=roll_var)
roll_entry.pack()

capture_button = tk.Button(root, text='Capture Images', command=capture_images_click)
capture_button.pack()

train_button = tk.Button(root, text='Train Model', command=train_model_click)
train_button.pack()

recognize_button = tk.Button(root, text='Recognize Face', command=recognize_face_click)
recognize_button.pack()

load_embeddings()

root.mainloop()

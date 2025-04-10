#!/usr/bin/env python
import os
import cv2
import time
import shutil
import threading
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key in production

# ----------------------------------
# A helper function to open the camera,
# trying multiple indexes with CAP_DSHOW first, then fallback
# ----------------------------------
def open_camera():
    for index in range(4):  # Try up to 4 camera indices
        # Try DirectShow on Windows (avoids many MSMF issues)
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"[INFO] Camera opened at index {index} with DirectShow.")
            return cap

        # If DirectShow fails, fallback to default backend
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"[INFO] Camera opened at index {index} with default backend.")
            return cap

    # If we get here, no camera opened
    print("[ERROR] No camera found or accessible on indexes 0..3.")
    return None


# ----------------------------------
# VideoCamera class: uses the above open_camera function
# ----------------------------------
class VideoCamera:
    def __init__(self):
        self.cap = open_camera()  # This may be None if no camera was found
        self.lock = threading.Lock()

    def __del__(self):
        if self.cap:
            self.cap.release()

    def get_frame(self):
        if not self.cap:
            return None
        with self.lock:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame


# Create a global camera instance
camera = VideoCamera()

# Global face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ----------------------------------
# Global "capture session" state
# ----------------------------------
capture_session = {
    "user": None,        # name of the person
    "target": 0,         # how many faces we want
    "count": 0,          # how many we have saved so far
    "active": False,     # capturing is on or off
    "last_save": 0.0,    # last time we saved an image
    "interval": 1.0      # min seconds between saves
}

def start_capture_session(user, target):
    capture_session["user"] = user
    capture_session["target"] = target
    capture_session["count"] = 0
    capture_session["active"] = True
    capture_session["last_save"] = time.time()

def stop_capture_session():
    capture_session["active"] = False


# ----------------------------------
# MJPEG generator for the "capture" page
# ----------------------------------
def gen_capture_feed():
    """
    Each frame:
      1. Convert to gray
      2. Detect faces (draw bounding boxes)
      3. If capturing is active and below target, automatically save ROI once per 'interval' seconds
      4. Render "Capturing X/Y" or "Capture Complete"
    """
    while True:
        frame = camera.get_frame()
        if frame is None:
            # If no frame, skip
            # Possibly camera is not open
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            text = "No camera found."
            cv2.putText(blank_frame, text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, jpeg = cv2.imencode('.jpg', blank_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.2)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Automatic face saving
        if capture_session["active"] and capture_session["count"] < capture_session["target"]:
            now = time.time()
            # Only save once every 'interval' seconds
            if (now - capture_session["last_save"]) >= capture_session["interval"]:
                if len(faces) > 0:
                    # Save first face found
                    (x, y, w, h) = faces[0]
                    face_roi = gray[y:y+h, x:x+w]
                    # Make dataset folder
                    user_dir = os.path.join("dataset", capture_session["user"])
                    os.makedirs(user_dir, exist_ok=True)
                    # Save
                    capture_session["count"] += 1
                    face_path = os.path.join(user_dir, f"{capture_session['count']}.jpg")
                    cv2.imwrite(face_path, face_roi)
                    capture_session["last_save"] = now

                    # If target reached
                    if capture_session["count"] >= capture_session["target"]:
                        stop_capture_session()

        # Draw bounding boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Overlay text
        if capture_session["active"]:
            overlay = f"Capturing {capture_session['count']}/{capture_session['target']}"
        else:
            if capture_session["count"] >= capture_session["target"] and capture_session["target"] > 0:
                overlay = "Capture Complete"
            else:
                overlay = "Not Capturing"

        cv2.putText(frame, overlay, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    """
    This feed is used by the capture page to display bounding boxes and 
    auto-capture faces on the server side.
    """
    return Response(gen_capture_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ----------------------------------
# Index/Home
# ----------------------------------
@app.route('/')
def index():
    return render_template('index.html')

# ----------------------------------
# Add User + Start Capture
# ----------------------------------
@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        name = request.form.get('name')
        target = int(request.form.get('target', 50))
        if not name:
            flash("Name is required.")
            return redirect(url_for('add_user'))
        # Start capturing
        start_capture_session(name, target)
        return redirect(url_for('capture'))
    return render_template('add_user.html')

@app.route('/capture')
def capture():
    return render_template('capture.html',
                           name=capture_session["user"],
                           target=capture_session["target"],
                           current_samples=capture_session["count"])

# ----------------------------------
# Train the LBPH model
# ----------------------------------
def train_model():
    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        print("No dataset found.")
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_mapping = {}
    current_label = 0

    for person_name in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_path):
            continue
        label_mapping[current_label] = person_name
        for filename in os.listdir(person_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(person_path, filename)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                faces.append(img)
                labels.append(current_label)
        current_label += 1

    if len(faces) == 0:
        print("No images found.")
        return False

    recognizer.train(faces, np.array(labels))
    recognizer.save("trainer.yml")
    np.save("label_mapping.npy", label_mapping)
    print("Training complete.")
    return True

@app.route('/train')
def train():
    if train_model():
        flash("Training complete.")
    else:
        flash("Training failed. No images available.")
    return redirect(url_for('index'))

# ----------------------------------
# Recognition feed
# ----------------------------------
def gen_recognition_feed():
    if not os.path.exists("trainer.yml") or not os.path.exists("label_mapping.npy"):
        while True:
            frame = camera.get_frame()
            if frame is None:
                # Show blank or error
                blank = np.zeros((480,640,3), dtype=np.uint8)
                cv2.putText(blank, "Model not trained.", (10,50),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                ret, jpeg = cv2.imencode('.jpg', blank)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                time.sleep(0.1)
                continue

            cv2.putText(frame, "Model not trained.", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.1)
    else:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("trainer.yml")
        label_mapping = np.load("label_mapping.npy", allow_pickle=True).item()

        while True:
            frame = camera.get_frame()
            if frame is None:
                # No frame, show black
                blank = np.zeros((480,640,3), dtype=np.uint8)
                cv2.putText(blank, "No camera found.", (10,50),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                ret, jpeg = cv2.imencode('.jpg', blank)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                time.sleep(0.1)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                try:
                    label, confidence = recognizer.predict(face_roi)
                    if confidence < 100:
                        name = label_mapping.get(label, "Unknown")
                        text = f"{name} ({int(confidence)})"
                    else:
                        text = "Unknown"
                except:
                    text = "Error"

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, text, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.1)

@app.route('/recognition_feed')
def recognition_feed():
    return Response(gen_recognition_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognize')
def recognize():
    return render_template('recognize.html')

# ----------------------------------
# Delete user
# ----------------------------------
@app.route('/delete_user')
def delete_user():
    dataset_dir = "dataset"
    users = []
    if os.path.exists(dataset_dir):
        users = [
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        ]
    return render_template('delete_user.html', users=users)

@app.route('/delete/<username>', methods=['POST'])
def delete(username):
    user_dir = os.path.join('dataset', username)
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)
        flash(f"User '{username}' deleted.")
        train_model()  # Retrain after deleting
    else:
        flash("User not found.")
    return redirect(url_for('delete_user'))

# ----------------------------------
# Run
# ----------------------------------
if __name__ == '__main__':
    app.run(debug=True)

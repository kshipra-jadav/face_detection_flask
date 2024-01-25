from flask import Flask, render_template, request, jsonify
import cv2
from mtcnn import MTCNN
import time
import concurrent.futures
import base64
import numpy as np

app = Flask(__name__)

def load_and_detect_haar(gray_img):
    print("inside haar")
    ff = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    ff_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    ff_alt2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    pf = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

    ff_faces = ff.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=10, minSize=(25, 25))
    ff_alt2_faces = ff_alt2.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=10, minSize=(20, 20))
    pf_faces = pf.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5, minSize=(20, 20))

    return ff_faces, ff_alt2_faces, pf_faces

def load_and_detect_mtcnn(image):
    print("inside mtcnn")
    mtcnn = MTCNN()
    faces = mtcnn.detect_faces(image)
    mt_faces = [face['box'] for face in faces]
    return mt_faces

def get_unique_face_locations(all_face_locations):
    unique_detected_faces = []
    for (x1, y1, w1, h1) in all_face_locations:
        unique = True
        for (x2, y2, w2, h2) in unique_detected_faces:
            if abs(x1 - x2) < 50 and abs(y1 - y2) < 50:
                unique = False
                break
        if unique:
            unique_detected_faces.append((x1, y1, w1, h1))
    
    return unique_detected_faces

@app.route('/')
def index():
    return render_template('pg1.html')

@app.route('/camera')
def camera():
    return render_template("pg2.html")

@app.route('/report')
def report():
    return render_template("pg4.html")

@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    filestr = request.files["img_upload"].read()
    file_bytes = np.fromstring(filestr, np.uint8) 
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        haar_detections = executor.submit(load_and_detect_haar, gray)
        mtcnn_detections = executor.submit(load_and_detect_mtcnn, img)
        
        ff_faces, ff_alt2_faces, pf_faces = haar_detections.result()
        mt_faces = mtcnn_detections.result()

        all_faces = [*ff_faces, *ff_alt2_faces, *pf_faces, *mt_faces]
        unique_detected_faces = get_unique_face_locations(all_faces)

        for (x, y, w, h) in unique_detected_faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        img = cv2.putText(img, f"{len(unique_detected_faces)} Faces", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 0), 10)

    _, buffer = cv2.imencode(".jpg", cv2.resize(img, (0, 0), fx=0.5, fy=0.5))
    b64 = base64.b64encode(buffer)

    return render_template("pg1.html", img_data = b64.decode('utf-8'))

if __name__ == '__main__':
    app.run(debug=True)

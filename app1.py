# from flask import Flask, render_template, Response
# import cv2
# from mtcnn import MTCNN
# import base64
# import numpy as np
# import concurrent.futures

# app = Flask(__name__)

# class FaceDetector:
#     def __init__(self):
#         self.detector = MTCNN()
#         self.face_count = 0

#     def detect_faces(self, frame):
#         faces = self.detector.detect_faces(frame)
#         self.face_count = len(faces)
#         return faces

# face_detector = FaceDetector()

# def generate_frames():
#     camera = cv2.VideoCapture(0)  # 0 indicates the default camera (you can change it to the camera index you want to use)

#     while True:
#         success, frame = camera.read()
#         if not success:
#             break

#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Perform face detection
#         faces = face_detector.detect_faces(frame)

#         for face in faces:
#             x, y, w, h = face['box']
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

#         # Display face count on each frame
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(frame, f'Faces: {face_detector.face_count}', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

#         _, buffer = cv2.imencode(".jpg", cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))
#         img_data = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + img_data + b'\r\n\r\n')

#     camera.release()

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/detect_faces')
# def detect_faces():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, Response, request
import cv2
from mtcnn import MTCNN
import numpy as np
import concurrent.futures

app = Flask(__name__)

class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()
        self.face_count = 0

    def detect_faces(self, frame):
        faces = self.detector.detect_faces(frame)
        self.face_count = len(faces)
        return faces

face_detector = FaceDetector()

def generate_frames(camera_index):
    camera = cv2.VideoCapture(camera_index)

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Perform face detection
        faces = face_detector.detect_faces(frame)

        for face in faces:
            x, y, w, h = face['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Display face count on each frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Faces: {face_detector.face_count}', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        _, buffer = cv2.imencode(".jpg", cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))
        img_data = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_data + b'\r\n\r\n')

    camera.release()

@app.route('/')
def index():
    return render_template('pg2.html')

@app.route('/detect_faces')
def detect_faces():
    camera_index = int(request.args.get('camera_index', 1))
    return Response(generate_frames(camera_index), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

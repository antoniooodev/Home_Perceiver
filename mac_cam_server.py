# mac_cam_server.py
import cv2
from flask import Flask, Response

app = Flask(__name__)
cap = cv2.VideoCapture(0)  # Mac’s webcam


def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        _, jpeg = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )


@app.route("/stream")
def stream():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

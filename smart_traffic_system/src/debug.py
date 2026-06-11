from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "data", "traffic.mp4")

print("Using:", VIDEO_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

ret, frame = cap.read()

if not ret:
    print("Could not read first frame")
    exit()

print("Frame shape:", frame.shape)

results = model(frame)

print(results[0])

annotated = results[0].plot()

cv2.imwrite("first_frame.jpg", frame)
cv2.imwrite("first_frame_detected.jpg", annotated)

print("Saved images")

cap.release()
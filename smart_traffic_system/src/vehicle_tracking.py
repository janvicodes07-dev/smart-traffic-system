from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "data", "traffic.mp4")

cap = cv2.VideoCapture(VIDEO_PATH)

cv2.namedWindow("Vehicle Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Vehicle Tracking", 1200, 800)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow(
        "Vehicle Tracking",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
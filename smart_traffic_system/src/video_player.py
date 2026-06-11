import cv2
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "smart_traffic_system", "data", "traffic.mp4")

print("Video:", VIDEO_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Could not open video")
    exit()

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("End of video")
        break

    frame_number += 1

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Traffic Video", frame)

    key = cv2.waitKey(30)

    if key == ord('q'):
        break

    if key == ord('n'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number + 100)

cap.release()
cv2.destroyAllWindows()
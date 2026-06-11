from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "data", "traffic.mp4")

print("Video Path:", VIDEO_PATH)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Creating resizable window
cv2.namedWindow("Vehicle Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Vehicle Detection", 1200, 800)

frame_count = 0

while True:

    # Reading frame
    ret, frame = cap.read()

    if not ret:
        print("End of video.")
        break

    frame_count += 1

    # Running YOLO detection
    results = model(frame, verbose=False)

    # Drawing bounding boxes
    annotated_frame = results[0].plot()

    # Resize for display
    display_frame = cv2.resize(
        annotated_frame,
        (1200, 800)
    )

    # Show frame number
    cv2.putText(
        display_frame,
        f"Frame: {frame_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display video
    cv2.imshow("Vehicle Detection", display_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
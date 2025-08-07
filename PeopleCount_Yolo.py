import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

st.title("People Counting with YOLOv8")
st.markdown("Upload a video to detect and count people in it.")

video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if video_file:
    # Save to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        count = 0

        for box in results[0].boxes:
            if int(box.cls[0]) == 0:  # Class 0 = person
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put count on frame
        cv2.putText(frame, f"People: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert color for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    cap.release()
    os.unlink(tfile.name)

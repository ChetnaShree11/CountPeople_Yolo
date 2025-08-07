#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install ultralytics opencv-python


# In[9]:


import cv2
from ultralytics import YOLO
from PIL import Image
from IPython.display import display, clear_output

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam or replace with a video file path like 'video.mp4'
cap = cv2.VideoCapture(0)  # Replace 0 with 'video.mp4' if needed

# Loop to capture frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Run inference
    results = model(frame)
    count = 0

    # Draw bounding boxes for 'person' class (class ID = 0)
    for box in results[0].boxes:
        if int(box.cls[0]) == 0:
            count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show count on image
    cv2.putText(frame, f"People: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert and display in Jupyter
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    clear_output(wait=True)
    display(Image.fromarray(img_rgb))
    
    # Optional: press 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    

# Release resources
cap.release()


# In[ ]:





# In[ ]:





# In[ ]:





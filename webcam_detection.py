import cv2
import torch
import numpy as np
from pathlib import Path

# Load YOLOv5 model (local)
model_path = Path("models/yolov5s.pt")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (YOLOv5 expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run detection
    results = model(frame_rgb)
    
    # Parse results
    detections = results.pandas().xyxy[0]  # DataFrame with columns: xmin, ymin, xmax, ymax, confidence, class, name
    
    # Draw bounding boxes
    for _, det in detections.iterrows():
        if det['confidence'] > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{det['name']} {det['confidence']:.2f}", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display
    cv2.imshow('YOLOv5 Webcam Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
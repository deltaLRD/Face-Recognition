from ultralytics import YOLO
import supervision as sv
import numpy as np
yolov8_model = YOLO("yolov8l.pt",task="detect")
import cv2
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    # cv2.imshow("Face-Recognition", frame)
    frame = np.asarray(frame)
    result = yolov8_model.predict(frame)[0]
    detections = sv.Detections.from_ultralytics(result)
    print(detections)
    break
    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()
cap.release()
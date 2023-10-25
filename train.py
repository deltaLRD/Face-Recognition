from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
model = YOLO("yolov8n.pt")
model.train(data="./datasets/data.yaml", epochs=2)
metrics = model.val()
model.export()
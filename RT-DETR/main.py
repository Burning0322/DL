#!uv pip install git+https://github.com/ultralytics/ultralytics@main
#!yolo checks

from ultralytics import RTDETR
import torch

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("rtdetr-l.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco.yaml", epochs=100, imgsz=640,batch = 64,optimizer="AdamW",lr0 = 0.001,patience=5)

# Run inference with the RT-DETR-l model on the 'bus.jpg' image
results = model("./kaggle/input/buspic/bus.jpg")
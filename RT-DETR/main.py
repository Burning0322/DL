from ultralytics import RTDETR
import torch

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print(device)

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("rtdetr-l.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=100, imgsz=640,batch=2)

# Run inference with the RT-DETR-l model on the 'bus.jpg' image
results = model("./bus.jpeg")
import torch
from torchvision import transforms
from PIL import Image
import json
from model import GoogLeNet

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

img_path = "Test4_GoogleNet/1.jpg"
img = Image.open(img_path)
img = data_transform(img)
img = torch.unsqueeze(img, dim=0)

json_path = "./Test4_GoogleNet/class_indices.json"
with open(json_path, "r") as f:
    class_indict = json.load(f)

weights_path = "./Test4_GoogleNet/GoogleNet.pth"

model = GoogLeNet(num_classes=5, aux_logits=True).to(device)

model.load_state_dict(torch.load(weights_path, map_location=device))

model.eval()

with torch.no_grad():
    output = torch.squeeze(model(img.to(device)))
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).cpu().numpy()  # Add .cpu()
print_res = "class: {}   prob: {:.3f}".format(class_indict[str(predict_cla)], predict[predict_cla].cpu().numpy())  # Add .cpu()
print(print_res)
for i in range(len(predict)):
    print("class: {:10}   prob: {:.3}".format(class_indict[str(i)], predict[i].cpu().numpy()))  # Add .cpu()
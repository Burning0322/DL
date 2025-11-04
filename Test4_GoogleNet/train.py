import torch
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from model import GoogLeNet

model = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
model.to(device)

if torch.cuda.device.count() > 1:
    print(f"Multiple GPUs detected: {torch.cuda.device_count()} GPUs will be used.")

    model = torch.nn.DataParallel(model)
model = model.to(device)

from torchvision import transforms

data_transform = {
    "train": 
        transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": 
        transforms.Compose([transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}

from torchvision import datasets

train_dataset = datasets.ImageFolder(root="/Users/renhonglow/Desktop/DL/flower_data/flower_photos/train", transform=data_transform["train"])
val_dataset = datasets.ImageFolder(root="/Users/renhonglow/Desktop/DL/flower_data/flower_photos/val", transform=data_transform["val"])


flower_list = train_dataset.class_to_idx
class_dict = dict((val,key) for key, val in flower_list.items())

json_str = json.dumps(class_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 4

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True, num_workers=0)
validate_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size, shuffle=False, num_workers=0)


loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 100
best_accurancy = 0.0
save_path = './GoogleNet.pth'

from tqdm import tqdm
import sys

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    train_bar = tqdm(train_loader, file=sys.stdout)

    for step,data in enumerate(train_bar):
        images,labels = data
        optimizer.zero_grad()
        logits,zux_logits2,aux_logits1 = model(images.to(device))
        loss0 = loss_function(logits, labels.to(device))
        loss1 = loss_function(aux_logits1, labels.to(device))
        loss2 = loss_function(zux_logits2, labels.to(device))
        loss = loss0 + loss1 * 0.3 +loss2 * 0.3
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}"

        model.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == val_labels.to(device)).sum().item()
            
            val_accurancy = acc / len(val_dataset)
            print(f'[epoch {epoch + 1}] val_accurancy: {val_accurancy:.3f}')
            if val_accurancy > best_accurancy:
                best_accurancy = val_accurancy
                torch.save(model.state_dict(), save_path)
    
    print("Finished Training")
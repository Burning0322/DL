import sys
import torch
from torchvision import transforms, datasets
import json
from model import resnet34
from torch import nn
import tqdm
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_transform = {
    "train":transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    ,
    "val":transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
}

train_dataset = datasets.ImageFolder(root="/Users/renhonglow/Desktop/DL/dataset/train",transform=data_transform["train"])

flower_list = train_dataset.class_to_idx

class_dict = dict((val,key) for key,val in flower_list.items())

with open("class_indices.json","w") as file:
    json.dump(class_dict,file,indent=4)

batch_size = 16

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

val_dataset = datasets.ImageFolder(root="/Users/renhonglow/Desktop/DL/dataset/val",transform=data_transform["val"])

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
)

print("using {} images for training, {} images for validation.".format(len(train_dataset),
                                                                        len(val_dataset)))

net = resnet34()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net.to(device)                      

model_weight_path = "/resnet34-pre.pth"

net.load_state_dict(torch.load(model_weight_path),map_location=device)

in_channel = net.fc.in_features
net.fc = nn.Linear(in_channel,5)
net.to(device)

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(),lr=0.0001)

epochs = 10
best_acc = 0.0
save = "./resNet34.pth"
train_steps = len(train_loader)

for epoch in range(epochs):
    net.train()
    running_loss= 0.0
    train_bar = tqdm(train_loader, file=sys.stdout)
    for step,data in enumerate(train_bar):
        images,labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        loss = loss_function(logits,labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch+1,epochs,loss)

    net.eval()
    acc = 0.0
    with torch.no_grad():
        val_bar = tqdm(val_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()

            val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)
        
    val_accurate = acc / len(val_dataset)
    print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
          (epoch + 1, running_loss / train_steps, val_accurate))
    
    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(net.state_dict(), save)

print('Finished Training')






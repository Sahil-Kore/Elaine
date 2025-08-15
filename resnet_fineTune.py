import os
import torch
import torch.nn as nn 
import torch.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    
])

val_transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

base_dataset = datasets.ImageFolder("./Rock Paper Scissors")

train_size=int(len(base_dataset) * 0.8)
val_size = len(base_dataset)- train_size

train_size = int(0.8 * len(base_dataset))
val_size = len(base_dataset) - train_size
train_indices, val_indices = random_split(range(len(base_dataset)), [train_size, val_size])

# Apply transforms separately using Subset
train_dataset = Subset(datasets.ImageFolder("./Rock Paper Scissors", transform=train_transform), train_indices)
val_dataset = Subset(datasets.ImageFolder("./Rock Paper Scissors", transform=val_transform), val_indices)

# DataLoaders
num_workers = os.cpu_count() - 4  # leave some CPU for system
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=num_workers)

class_names=base_dataset.classes
idx_to_classes = {v:k for k,v in base_dataset.class_to_idx.items()}

device ="cuda" if torch.cuda.is_available() else "cpu"

def imshow(img:torch.Tensor, label = None):
    inp =img.permute(1,2,0)
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    inp= inp * std +mean
    inp = np.clip(inp,0 ,1)
    plt.imshow(inp)
    if label:
        plt.title(idx_to_classes[label])
    plt.pause(0.001)


count=1
for image,label in train_dataset:
    imshow(image,label)
    if count==10:break
    count+=1


model =models.resnet34(pretrained = True)
summary(model, input_size=(1, 3, 224, 224),verbose=2)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

count_parameters(model)
model.parameters
model.fc
model.fc= nn.Linear(in_features=512,out_features= len(class_names))

save_loss = {'train': [], 'test': []}
save_acc = {'train': [], 'test': []}

for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

torch.set_float32_matmul_precision("high")  
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-4)
model = torch.compile(model)  # optional, only if PyTorch 2.0+
#Training loop
for epoch in range(20):
    # --- TRAIN ---
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    epoch_train_loss = running_loss / len(train_loader.dataset)
    epoch_train_acc = correct_train / total_train

    # --- VALIDATION ---
    model.eval()
    running_loss_val = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss_val += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

    epoch_val_loss = running_loss_val / len(val_loader.dataset)
    epoch_val_acc = correct_val / total_val

    # --- Save metrics ---
    save_loss['train'].append(epoch_train_loss)
    save_loss['test'].append(epoch_val_loss)
    save_acc['train'].append(epoch_train_acc)
    save_acc['test'].append(epoch_val_acc)

    print(f"Epoch [{epoch+1}/20] "
          f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} "
          f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
    
if hasattr(model , "_orig_mod"):
    print(True)
    model_to_save = model._orig_mod
else:
    model_to_save =model
    
torch.save(model_to_save.state_dict(),"./Models/models.pth")
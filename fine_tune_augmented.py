import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import io
from torchvision import datasets, transforms, models
from torch.utils.data import Subset
from torch.utils.data import DataLoader, random_split
import os
checkpoint = torch.load('./Models/model_bundle.pth',map_location="cpu", weights_only=False)
model = models.resnet34(pretrained=False)
model.fc = nn.Linear(model.fc.in_features , 3)

state_dict = checkpoint["model_state_dict"]
model.load_state_dict(state_dict)
transform = checkpoint["transform"]
idx_to_classes= checkpoint["idx_to_classes"]

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((256,256)),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    
])

val_transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

base_dataset = datasets.ImageFolder("./RockPaperScissors2/versions/2")

train_size=int(len(base_dataset) * 0.8)
val_size = len(base_dataset)- train_size

train_size = int(0.8 * len(base_dataset))
val_size = len(base_dataset) - train_size
train_indices, val_indices = random_split(range(len(base_dataset)), [train_size, val_size])

# Apply transforms separately using Subset
train_dataset = Subset(datasets.ImageFolder("./RockPaperScissors2/versions/2", transform=train_transform), train_indices)
val_dataset = Subset(datasets.ImageFolder("./RockPaperScissors2/versions/2", transform=val_transform), val_indices)

# DataLoaders
num_workers = os.cpu_count() - 4  # leave some CPU for system
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=num_workers)


save_loss = {'train': [], 'test': []}
save_acc = {'train': [], 'test': []}

for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")  
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-4)
model = torch.compile(model)  # optional, only if PyTorch 2.0+
epochs =20
#Training loop
for epoch in range(epochs):
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

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} "
          f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
    
if hasattr(model , "_orig_mod"):
    print(True)
    model_to_save = model._orig_mod
else:
    model_to_save =model


torch.save({
    "model_state_dict":model_to_save.state_dict(),
    "transform": val_transform,
    "idx_to_classes":idx_to_classes
    },"./Models/model_bundle_kaggle2.pth")    

    

from PIL import Image
img_dir = "./Test"
model.eval()
with torch.no_grad():
        for file_name in os.listdir(img_dir):
            file_path = os.path.join(img_dir, file_name)

            # Skip non-image files just in case
            if not file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                continue

            image = Image.open(file_path).convert("RGB")
            pred = model(val_transform(image).unsqueeze(0).to(device))
            
            predicted_class = pred[0].argmax().item()
            print(f"{file_name} -> class {predicted_class}")

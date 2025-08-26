import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import transforms
from PIL import Image
model = models.resnet34(pretrained=False)
model.fc = nn.Linear(model.fc.in_features , 3)

device ="cuda"  if torch.cuda.is_available() else "cpu"
state_dict = torch.load("./Models/models.pth", map_location= torch.device(device))

model.load_state_dict(state_dict)

val_transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

inputs = [ val_transform(Image.open(img).convert("RGB")) for img in ["./Test/30c607dc-c35e-484f-a761-639be36b74c0.jpg","./Test/581878cd-621a-454f-993e-225372d84448.jpg","./Test/7658f2c9-fee0-4887-9853-aa54d733da15.jpg"]]

idx_to_classes ={
    0:"Paper",
    1:"Rock",
    2:"Scissors"
}

img_path = "./Test/7658f2c9-fee0-4887-9853-aa54d733da15.jpg"
image = Image.open(img_path).convert("RGB") 

pred = model(val_transform(image).unsqueeze(0))
pred
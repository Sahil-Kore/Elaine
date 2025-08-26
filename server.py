import torch
import torch.nn as nn
from torchvision import models
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
checkpoint = torch.load('./Models/model_bundle_kaggle2.0.pth',map_location="cpu", weights_only=False)
model = models.resnet34(pretrained=False)
model.fc = nn.Linear(model.fc.in_features , 3)

state_dict = checkpoint["model_state_dict"]
model.load_state_dict(state_dict)
transform = checkpoint["transform"]
idx_to_classes= checkpoint["idx_to_classes"]


app= FastAPI()

@app.get("/")
def reed_root():
    return {'message' : "RPC Model Api"}

@app.post("/predict")
async def predict(file:UploadFile = File(...)):
   #read image data
   image_bytes =  await file.read()
   
   image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
   input_img = transform(image)
   prediction= model(input_img.unsqueeze(0))
   prediction = prediction[0].argmax(0)
   result = idx_to_classes[prediction]
   return JSONResponse({
       "result":result
   })

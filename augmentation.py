import os
from PIL import Image
from torchvision import transforms
import random

# Paths
input_dir = "./RockPaperScissors"   # has subfolders like rock/, paper/, scissors/
output_dir = "./dataset_augmented"

# Augmentation transform (can make it aggressive for small datasets)
augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
])

# Create output dir
os.makedirs(output_dir, exist_ok=True)

# Loop over classes
for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    
    output_class_path = os.path.join(output_dir, class_name)
    os.makedirs(output_class_path, exist_ok=True)
    
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            continue  # skip broken files

        # Save original
        image.save(os.path.join(output_class_path, img_name))
        
        # Create N augmentations
        for i in range(5):  # change 5 → how many extra you want
            aug_img = augment_transform(image)
            aug_img.save(os.path.join(output_class_path, f"{img_name.split('.')[0]}_aug{i}.jpg"))

print("✅ Augmentation complete!")

import argparse
import os
from PIL import Image
import torch
from torchvision import models, transforms
from torchvision.models import vgg16, resnet18, alexnet, VGG16_Weights, ResNet18_Weights, AlexNet_Weights

# ----------------------------
# Step 1: Parse command-line arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='pet_images/', help='folder with images')
parser.add_argument('--arch', type=str, default='vgg', help='model architecture: vgg, resnet, alexnet')
parser.add_argument('--dogfile', type=str, default='dognames.txt', help='text file with dog breed names')
args = parser.parse_args()

# ----------------------------
# Step 2: Load Pretrained Model
# ----------------------------
if args.arch == 'vgg':
    model = vgg16(weights=VGG16_Weights.DEFAULT)
elif args.arch == 'resnet':
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
elif args.arch == 'alexnet':
    model = alexnet(weights=AlexNet_Weights.DEFAULT)
else:
    print("Invalid architecture! Choose from: vgg, resnet, alexnet")
    exit()

model.eval()

# ----------------------------
# Step 3: Transform images
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ----------------------------
# Step 4: Load dog names from file
# ----------------------------
with open(args.dogfile, 'r') as f:
    dog_names = [line.strip().lower() for line in f.readlines()]

# ----------------------------
# Step 5: Go through each image and classify
# ----------------------------
correct_dog = 0
total_images = 0

for filename in os.listdir(args.dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(args.dir, filename)
        image = Image.open(image_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
        _, predicted = output.max(1)
        class_id = predicted.item()

        # Load class names from ImageNet
        label_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        import requests
        labels = requests.get(label_url).text.split("\n")
        label = labels[class_id].lower()

        is_dog = any(breed in label for breed in dog_names)

        print(f"Image: {filename} | Predicted: {label} | Is Dog: {is_dog}")

        if is_dog:
            correct_dog += 1
        total_images += 1

# ----------------------------
# Step 6: Print Summary
# ----------------------------
print("\n--- Summary ---")
print(f"Total Images: {total_images}")
print(f"Dog Images Detected: {correct_dog}")
print(f"Accuracy: {(correct_dog / total_images) * 100:.2f}%")

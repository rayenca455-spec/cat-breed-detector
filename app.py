import gradio as gr
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import json

with open("breeds.json") as f:
    breeds = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(breeds))
model.load_state_dict(torch.load("cat_breed_model.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        top3 = probabilities.topk(3)

    results = {}
    for i in range(3):
        breed = breeds[top3.indices[i].item()].replace("_", " ").title()
        confidence = round(top3.values[i].item() * 100, 2)
        results[breed] = confidence / 100

    return results

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a photo of your cat or dog"),
    outputs=gr.Label(num_top_classes=3, label="Breed Prediction"),
    title="🐾 Pet Breed Detector",
    description="Upload a photo of your cat or dog and the AI will identify the breed. Supports 12 cat breeds and 25 dog breeds.",
    examples=[],
    theme=gr.themes.Soft()
)

demo.launch()

import torch
import gradio as gr
from torch import nn
import requests
import io

# Fetch class names from URL
class_names_url = "https://huggingface.co/spaces/course-demos/Sketch-Recognition/raw/main/class_names.txt"
print("Fetching class names from URL...")
response = requests.get(class_names_url)
LABELS = response.text.strip().split()

print(f"Loaded {len(LABELS)} class names from URL")
print(f"First 10 class names: {LABELS[:10]}")

model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(1152, 256),
    nn.ReLU(),
    nn.Linear(256, len(LABELS)),
)

# Fetch model weights from URL
model_url = "https://huggingface.co/spaces/course-demos/Sketch-Recognition/resolve/main/pytorch_model.bin"
print("Downloading model weights from URL...")
response = requests.get(model_url)
state_dict = torch.load(io.BytesIO(response.content), map_location="cpu")
model.load_state_dict(state_dict, strict=False)
print("Model weights loaded successfully from URL")

model.eval()


def predict(im):
    x = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        out = model(x)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    values, indices = torch.topk(probabilities, 5)
    return {LABELS[i]: v.item() for i, v in zip(indices, values)}


# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", image_mode="L"),
    outputs=gr.Label(num_top_classes=5),
    title="Sketch Recognition",
    description="Draw something and the model will predict what it is!",
    examples=[
        ["examples/cat.jpg"],
        ["examples/car.jpg"],
        ["examples/house.jpg"]
    ]
)

if __name__ == "__main__":
    print("Starting Gradio interface...")
    iface.launch() 
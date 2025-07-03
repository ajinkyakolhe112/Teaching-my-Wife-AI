# Converted from 4 - live image classification.ipynb - Markdown format optimized for LLM readability

```python
from pathlib import Path
import torch
import gradio as gr
from torch import nn
import requests
import io

# Fetch class names from URL
url = "https://huggingface.co/spaces/course-demos/Sketch-Recognition/raw/main/class_names.txt"
response = requests.get(url)
LABELS = response.text.strip().split()

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
model_url  = "https://huggingface.co/spaces/course-demos/Sketch-Recognition/resolve/main/pytorch_model.bin"
response   = requests.get(model_url)
state_dict = torch.load(io.BytesIO(response.content), map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()


def predict(im):
    x = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        out = model(x)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    values, indices = torch.topk(probabilities, 5)
    return {LABELS[i]: v.item() for i, v in zip(indices, values)}
```

# Bugs corrected with help of colab gemini ai
- `theme="gradio/default"`
- 

```python
interface = gr.Interface(
    predict,
    inputs="sketchpad",
    outputs="label",
    theme="gradio/default", 
    title="Sketch Recognition",
    description="Who wants to play Pictionary? Draw a common object like a shovel or a laptop, and the algorithm will guess in real time!",
    article="<p style='text-align: center'>Sketch Recognition | Demo Model</p>",
    live=True,
)
interface.launch(share=True)
```

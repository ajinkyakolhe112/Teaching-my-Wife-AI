# Converted from 2 - chatbot model in browser.ipynb - Markdown format optimized for LLM readability

```python
from transformers import pipeline

model = pipeline("text-generation")


def predict(prompt):
    completion = model(prompt)[0]["generated_text"]
    return completion

predict("My favorite programming language is")
```

```python
import gradio as gr

gr.Interface(fn=predict, inputs="text", outputs="text").launch()
```

```python

```

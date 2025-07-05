# Converted from week5_lecture1_hf_pipeline_sentiment.ipynb - Markdown format optimized for LLM readability

# Lecture 5.1: Sentiment Analysis with Hugging Face Pipelines and Pre-trained Models

## Introduction

The Hugging Face `transformers` library offers a high-level abstraction called `pipeline` that makes it incredibly easy to use pre-trained models for various NLP tasks, including sentiment analysis. Pipelines encapsulate the entire process: loading a pre-trained model and its tokenizer, pre-processing input text, running the text through the model, and post-processing the model's output into a human-readable format.

This notebook will demonstrate how to use the `sentiment-analysis` pipeline, explore its components, and understand what happens under the hood.

### 1. Basic Usage of the Sentiment Analysis Pipeline

Using the `pipeline` function for sentiment analysis is straightforward. If you don't specify a model, it defaults to a pre-trained model fine-tuned for sentiment analysis (often DistilBERT SST-2).

```python
# Ensure transformers library is installed
# !pip install transformers

from transformers import pipeline

# Initialize the sentiment analysis pipeline
# If no model is specified, it uses a default model for this task.
# The first time you run this, it will download the default model and tokenizer.
classifier = pipeline(task="sentiment-analysis")

# Prepare some example texts
texts = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
    "The movie was okay, not great but not terrible either.",
    "This is an absolutely fantastic experience!"
]

# Get predictions
predictions = classifier(texts)

# Print the predictions
for text, prediction in zip(texts, predictions):
    print(f"Text: {text}")
    print(f"Label: {prediction['label']}, Score: {prediction['score']:.4f}\n")
```

### 2. Specifying a Model for the Pipeline

You can choose a specific model from the Hugging Face Hub to use within the pipeline. For example, `nlptown/bert-base-multilingual-uncased-sentiment` is a model fine-tuned for sentiment analysis on product reviews in multiple languages, and it can output ratings from 1 to 5 stars.

```python
from transformers import pipeline

# Specify a different model for sentiment analysis
# This model predicts star ratings (1 to 5 stars)
specific_classifier = pipeline(
    task="sentiment-analysis", 
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

reviews = [
    "This product is amazing, exceeded all my expectations!", # English
    "C'est un désastre complet, ne l'achetez pas.", # French (This is a complete disaster, don't buy it.)
    "El servicio fue terrible y la comida peor.", # Spanish (The service was terrible and the food worse.)
    "Ganz gut, aber nichts besonderes.", # German (Quite good, but nothing special.)
    "まあまあです" # Japanese (It's so-so / okay)
]

star_predictions = specific_classifier(reviews)

for review, prediction in zip(reviews, star_predictions):
    print(f"Review: {review}")
    print(f"Predicted Rating: {prediction['label']}, Score: {prediction['score']:.4f}\n")
```

### 3. Exploring the Pipeline Object

A pipeline object (like our `classifier` or `specific_classifier`) bundles together the model, tokenizer, and other configuration. Let's inspect the components of the default classifier.

```python
# Using the default classifier from the first example
print(f"Task: {classifier.task}") 
print(f"Model: {type(classifier.model)}")
print(f"Tokenizer: {type(classifier.tokenizer)}")
print(f"Device: {classifier.device}")

# You can print the model itself to see its architecture
# print("\nModel Architecture:")
# print(classifier.model)

# And the tokenizer configuration
# print("\nTokenizer Configuration:")
# print(classifier.tokenizer)
```

The default model for `sentiment-analysis` is often `distilbert-base-uncased-finetuned-sst-2-english`. 
This model has:
*   A **tokenizer** (`DistilBertTokenizerFast`) to convert text into numerical IDs.
*   A **model** (`DistilBertForSequenceClassification`) which is a DistilBERT transformer architecture with a sequence classification head on top.

Let's break down the process manually.

#### 3.1. Tokenization

The tokenizer first converts text into tokens (words, sub-words, or symbols) and then maps these tokens to numerical IDs. It also adds special tokens required by the model (like `[CLS]` and `[SEP]`).

```python
from transformers import AutoTokenizer

# Load the tokenizer used by the default sentiment analysis pipeline
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english" # Default model checkpoint
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

# Tokenize the inputs
# padding=True ensures all sequences in a batch have the same length.
# truncation=True cuts sequences longer than the model's max length.
# return_tensors="pt" returns PyTorch tensors.
tokenized_inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")

print("Tokenized Inputs:")
for key, value in tokenized_inputs.items():
    print(f"{key}: {value}")

print(f"\nShape of input_ids: {tokenized_inputs['input_ids'].shape}")

# You can decode the input_ids back to tokens to see what they look like
print(f"\nTokens for the first sentence: {tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'][0])}")
```

#### 3.2. Model Inference

The tokenized inputs are then passed to the model. The model outputs "logits," which are raw, unnormalized scores for each class (e.g., positive and negative).

```python
from transformers import AutoModelForSequenceClassification
import torch

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# Move inputs to the same device as the model (e.g., GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs_on_device = {k: v.to(device) for k, v in tokenized_inputs.items()}

# Get model outputs (logits)
with torch.no_grad(): # Disable gradient calculations for inference
    outputs = model(**inputs_on_device)

print("Model Outputs (Logits):")
print(outputs.logits)
print(f"\nShape of logits: {outputs.logits.shape}") # (batch_size, num_labels)
```

The output `logits` are raw scores. For the first sentence ("I've been waiting..."), the logits might be `[-1.5607,  1.6123]`. For the second ("I hate this..."), they might be `[ 3.1931, -2.6685]`.

These represent the model's confidence for 'NEGATIVE' (first value) and 'POSITIVE' (second value) respectively, before normalization.

#### 3.3. Post-processing (Softmax)

To convert these logits into probabilities, a Softmax function is applied. The pipeline does this automatically.

```python
import torch.nn.functional as F

# Apply softmax to convert logits to probabilities
probabilities = F.softmax(outputs.logits, dim=-1)
print("Probabilities:")
print(probabilities)

# Get the predicted label ID (0 for 'NEGATIVE', 1 for 'POSITIVE' in this model's config)
predicted_label_ids = torch.argmax(probabilities, dim=-1)
print(f"\nPredicted Label IDs: {predicted_label_ids}")

# Map label IDs to human-readable labels (from the model's configuration)
labels = [model.config.id2label[label_id] for label_id in predicted_label_ids.tolist()]
print(f"Predicted Labels: {labels}")

# The pipeline also returns the score of the predicted class
scores = torch.max(probabilities, dim=-1).values
print(f"Scores for predicted labels: {scores}")
```

This manual breakdown shows the steps that the `pipeline` function handles automatically:
1.  **Tokenization:** Text to input IDs.
2.  **Inference:** Input IDs to logits.
3.  **Post-processing:** Logits to probabilities and then to human-readable labels with scores.

This makes pipelines very convenient for quick application of pre-trained models.

## Conclusion

Hugging Face Pipelines provide a very simple and effective way to use pre-trained models for a variety of NLP tasks, including sentiment analysis. They abstract away much of the boilerplate code for loading models, tokenizing text, performing inference, and processing the results.

Key Takeaways:
*   Pipelines are great for **quick prototyping** and applying models with minimal code.
*   You can use the **default model** for a task or specify any compatible model from the Hugging Face Hub.
*   Under the hood, pipelines perform **tokenization, model inference (getting logits), and post-processing (like softmax and label mapping)**.
*   Understanding these underlying steps can be helpful when you need more control or want to build custom model interaction logic.

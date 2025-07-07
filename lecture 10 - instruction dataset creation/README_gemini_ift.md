# Simple IFT Dataset Creation with Gemini

This script creates instruction fine-tuning datasets using Gemini directly for RAG, eliminating the need for local vector databases and complex embeddings.

## Features

- ✅ Uses Gemini for both RAG and dataset generation
- ✅ Simple, easy-to-understand code structure  
- ✅ Creates diverse question-answer pairs
- ✅ Saves datasets in multiple formats (JSON and Alpaca format)
- ✅ No complex dependencies or local vector databases

## Setup

1. Install the required package:
```bash
pip install -r requirements_gemini_ift.txt
```

2. Make sure you have the Pride and Prejudice text file:
```
datasets/pride_prejudice.txt
```

## Usage

Simply run the script:
```bash
python 3_ift_dataset_with_gemini.py
```

## What it does

1. **Loads the book text** from `datasets/pride_prejudice.txt`
2. **Generates diverse questions** about Pride and Prejudice using Gemini
3. **Creates RAG prompts** that include the book text and each question
4. **Generates detailed answers** using Gemini's knowledge of the text
5. **Saves the dataset** in two formats:
   - `pride_prejudice_ift_dataset.json` - Standard format
   - `pride_prejudice_alpaca.json` - Alpaca format for fine-tuning

## Output

The script will create:
- A dataset with question-answer pairs
- Preview of the first few samples
- Two JSON files in different formats

## Customization

You can easily modify:
- Number of questions (change `num_questions` parameter)
- Book text source (modify `_load_pride_prejudice` method)
- Question types (modify the prompt in `generate_questions`)
- Output formats (add new save methods)

## Advantages over traditional RAG

- **Simpler**: No need for embeddings, vector databases, or chunking
- **Faster**: Direct API calls to Gemini
- **More reliable**: Uses Gemini's built-in knowledge and reasoning
- **Easier to understand**: Clear, linear code flow
- **No local resources**: No need to download models or create embeddings 
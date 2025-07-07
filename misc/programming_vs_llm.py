from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import re

app = Flask(__name__)

# Initialize the local LLM for word meaning
word_meaning_model = pipeline("text2text-generation", model="google/flan-t5-small")

def count_words(text):
    # Split text into words and filter out empty strings
    words = [word for word in re.findall(r'\w+', text.lower()) if word]
    return len(words)

def get_word_meaning(word):
    # Create a prompt for the model
    prompt = f"What is the detailed meaning of the word '{word}'?"
    
    # Generate meaning using the local LLM
    response = word_meaning_model(prompt, max_length=50, num_return_sequences=1)
    return response[0]['generated_text']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/count-words', methods=['POST'])
def count_words_endpoint():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    word_count = count_words(text)
    return jsonify({'word_count': word_count})

@app.route('/word-meanings', methods=['POST'])
def word_meanings_endpoint():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    words = [word for word in re.findall(r'\w+', text.lower()) if word]
    
    meanings = {}
    for word in words:
        meanings[word] = get_word_meaning(word)
    
    return jsonify({'meanings': meanings})

if __name__ == '__main__':
    app.run(debug=True)
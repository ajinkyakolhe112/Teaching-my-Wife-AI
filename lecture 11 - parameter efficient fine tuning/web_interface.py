#!/usr/bin/env python3
"""
Web Interface for Pride & Prejudice Fine-tuned Model
Lecture 11 - Parameter Efficient Fine-tuning

A simple Flask web application to interact with the fine-tuned model.
"""

import json
import torch
from flask import Flask, request, jsonify, render_template_string
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model(model_path: str):
    """Load the fine-tuned model and tokenizer"""
    global model, tokenizer
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    
    # Load model info
    model_info_path = os.path.join(model_path, "model_info.json")
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        print(f"Model info: {model_info}")
    
    return model, tokenizer

def generate_response(instruction: str, max_length: int = 200, temperature: float = 0.7) -> str:
    """Generate a response using the fine-tuned model"""
    
    if model is None or tokenizer is None:
        raise ValueError("Model not loaded")
    
    # Format input
    input_text = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response part
    response = response.split("### Response:\n")[-1].strip()
    
    return response

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pride & Prejudice AI Assistant</title>
    <style>
        body {
            font-family: 'Georgia', serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f6f1;
            color: #2c1810;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #8b4513;
            padding-bottom: 20px;
        }
        .header h1 {
            color: #8b4513;
            font-size: 2.5em;
            margin: 0;
        }
        .header p {
            color: #666;
            font-style: italic;
            margin: 10px 0 0 0;
        }
        .chat-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .input-section {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .input-section input[type="text"] {
            flex: 1;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            font-family: inherit;
        }
        .input-section button {
            padding: 12px 24px;
            background-color: #8b4513;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
        }
        .input-section button:hover {
            background-color: #a0522d;
        }
        .controls {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            align-items: center;
        }
        .control-group {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .control-group label {
            font-weight: bold;
            color: #666;
        }
        .control-group input[type="range"] {
            width: 100px;
        }
        .control-group input[type="number"] {
            width: 60px;
            padding: 5px;
            border: 1px solid #ddd;
            border-radius: 3px;
        }
        .response-section {
            background: #f9f9f9;
            border-left: 4px solid #8b4513;
            padding: 15px;
            border-radius: 5px;
            min-height: 100px;
        }
        .response-section h3 {
            margin-top: 0;
            color: #8b4513;
        }
        .loading {
            color: #666;
            font-style: italic;
        }
        .error {
            color: #d32f2f;
            background: #ffebee;
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid #d32f2f;
        }
        .example-questions {
            background: #e8f5e8;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .example-questions h3 {
            margin-top: 0;
            color: #2e7d32;
        }
        .example-questions ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        .example-questions li {
            margin: 5px 0;
            cursor: pointer;
            color: #1976d2;
        }
        .example-questions li:hover {
            text-decoration: underline;
        }
        .model-info {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
            font-size: 14px;
        }
        .model-info h3 {
            margin-top: 0;
            color: #1565c0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Pride & Prejudice AI Assistant</h1>
        <p>Ask questions about Jane Austen's beloved novel</p>
    </div>
    
    <div class="chat-container">
        <div class="controls">
            <div class="control-group">
                <label for="temperature">Temperature:</label>
                <input type="range" id="temperature" min="0.1" max="1.0" step="0.1" value="0.7">
                <input type="number" id="tempValue" min="0.1" max="1.0" step="0.1" value="0.7" style="width: 50px;">
            </div>
            <div class="control-group">
                <label for="maxLength">Max Length:</label>
                <input type="number" id="maxLength" min="50" max="500" value="200">
            </div>
        </div>
        
        <div class="input-section">
            <input type="text" id="questionInput" placeholder="Ask a question about Pride & Prejudice..." onkeypress="handleKeyPress(event)">
            <button onclick="askQuestion()">Ask</button>
        </div>
        
        <div class="response-section">
            <h3>Response:</h3>
            <div id="responseText">Enter a question above to get started...</div>
        </div>
    </div>
    
    <div class="example-questions">
        <h3>Example Questions:</h3>
        <ul>
            <li onclick="setQuestion('What is Elizabeth Bennet\'s personality like?')">What is Elizabeth Bennet's personality like?</li>
            <li onclick="setQuestion('How does Mr. Darcy change throughout the novel?')">How does Mr. Darcy change throughout the novel?</li>
            <li onclick="setQuestion('What is the significance of the first proposal scene?')">What is the significance of the first proposal scene?</li>
            <li onclick="setQuestion('Describe the relationship between Jane and Mr. Bingley.')">Describe the relationship between Jane and Mr. Bingley.</li>
            <li onclick="setQuestion('What role does Mr. Wickham play in the story?')">What role does Mr. Wickham play in the story?</li>
            <li onclick="setQuestion('How does the social structure of Regency England influence the plot?')">How does the social structure of Regency England influence the plot?</li>
        </ul>
    </div>
    
    <div class="model-info">
        <h3>Model Information:</h3>
        <p><strong>Base Model:</strong> <span id="baseModel">Loading...</span></p>
        <p><strong>Fine-tuning Method:</strong> <span id="fineTuningMethod">Loading...</span></p>
        <p><strong>Description:</strong> <span id="description">Loading...</span></p>
    </div>

    <script>
        // Sync temperature slider and input
        document.getElementById('temperature').addEventListener('input', function() {
            document.getElementById('tempValue').value = this.value;
        });
        
        document.getElementById('tempValue').addEventListener('input', function() {
            document.getElementById('temperature').value = this.value;
        });
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                askQuestion();
            }
        }
        
        function setQuestion(question) {
            document.getElementById('questionInput').value = question;
        }
        
        async function askQuestion() {
            const question = document.getElementById('questionInput').value.trim();
            if (!question) return;
            
            const temperature = parseFloat(document.getElementById('temperature').value);
            const maxLength = parseInt(document.getElementById('maxLength').value);
            
            const responseText = document.getElementById('responseText');
            responseText.innerHTML = '<span class="loading">Thinking...</span>';
            
            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        question: question,
                        temperature: temperature,
                        max_length: maxLength
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    responseText.innerHTML = data.response.replace(/\\n/g, '<br>');
                } else {
                    responseText.innerHTML = `<span class="error">Error: ${data.error}</span>`;
                }
            } catch (error) {
                responseText.innerHTML = `<span class="error">Error: ${error.message}</span>`;
            }
        }
        
        // Load model info on page load
        window.addEventListener('load', async function() {
            try {
                const response = await fetch('/model_info');
                const data = await response.json();
                
                document.getElementById('baseModel').textContent = data.base_model || 'Unknown';
                document.getElementById('fineTuningMethod').textContent = data.fine_tuning_method || 'Unknown';
                document.getElementById('description').textContent = data.description || 'Unknown';
            } catch (error) {
                console.error('Error loading model info:', error);
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Home page with the web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/ask', methods=['POST'])
def ask_question():
    """API endpoint for asking questions"""
    
    try:
        data = request.get_json()
        question = data.get('question', '')
        temperature = data.get('temperature', 0.7)
        max_length = data.get('max_length', 200)
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        response = generate_response(question, max_length, temperature)
        
        return jsonify({
            "question": question,
            "response": response,
            "temperature": temperature,
            "max_length": max_length
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get model information"""
    
    model_info_path = os.path.join(app.config.get('MODEL_PATH', './pride_prejudice_model'), 'model_info.json')
    
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        return jsonify(model_info)
    else:
        return jsonify({
            "base_model": "Unknown",
            "fine_tuning_method": "Unknown",
            "description": "Model information not available"
        })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    })

def main():
    """Main function to run the web interface"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Pride & Prejudice AI Assistant Web Interface')
    parser.add_argument('--model_path', type=str, default='./pride_prejudice_model',
                       help='Path to the fine-tuned model')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to bind the server to')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Set model path in app config
    app.config['MODEL_PATH'] = args.model_path
    
    try:
        # Load the model
        load_model(args.model_path)
        
        print(f"Starting web interface on http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop the server")
        
        # Run the Flask app
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
        
    except Exception as e:
        print(f"Error starting web interface: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 
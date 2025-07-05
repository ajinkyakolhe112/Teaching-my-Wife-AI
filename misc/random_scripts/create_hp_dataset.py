import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import os

# Step 1: Define our dataset creator class
class HarryPotterDatasetCreator:
    def __init__(self):
        print("Loading the language model...")
        # Load a small model that's easy to run
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        self.model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
        print("Model loaded successfully!")

        # Define what types of questions we want to generate
        self.instruction_types = [
            "Character Analysis",
            "Plot Explanation",
            "Spell Information",
            "Location Description"
        ]

        # Sample data for each category
        self.sample_data = {
            "character": ["Harry Potter", "Hermione Granger", "Ron Weasley"],
            "event": ["The Battle of Hogwarts", "The Triwizard Tournament"],
            "spell": ["Expelliarmus", "Wingardium Leviosa"],
            "location": ["Hogwarts", "Diagon Alley"]
        }

    def generate_question_answer(self, category):
        """Generate a single question-answer pair"""
        # Get a random item from the selected category
        item = random.choice(self.sample_data[category.lower().split()[0]])
        
        # Create a prompt for the model
        prompt = f"Create a question and answer about {item} from Harry Potter. Format as 'Question: [question] Answer: [answer]'"
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=200)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=300,
            temperature=0.7,
            num_return_sequences=1
        )
        
        # Get the generated text
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Try to split into question and answer
        try:
            question, answer = response.split("Answer:", 1)
            question = question.replace("Question:", "").strip()
            answer = answer.strip()
        except:
            question = prompt
            answer = response
        
        return {
            "instruction": question,
            "response": answer,
            "type": category
        }

    def create_dataset(self, num_samples=5):
        """Create a small dataset of question-answer pairs"""
        dataset = []
        
        print(f"\nGenerating {num_samples} samples...")
        for _ in range(num_samples):
            # Pick a random category
            category = random.choice(self.instruction_types)
            # Generate a question-answer pair
            sample = self.generate_question_answer(category)
            dataset.append(sample)
            print(f"Generated {category} sample")
        
        return dataset

    def save_dataset(self, dataset, filename="harry_potter_qa.json"):
        """Save the dataset to a file"""
        # Create datasets directory if it doesn't exist
        os.makedirs("datasets", exist_ok=True)
        
        # Save the file
        filepath = os.path.join("datasets", filename)
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"\nDataset saved to: {filepath}")

def main():
    print("="*50)
    print("Harry Potter Question-Answer Dataset Creator")
    print("="*50)
    
    # Create our dataset creator
    creator = HarryPotterDatasetCreator()
    
    # Generate a small dataset
    dataset = creator.create_dataset(num_samples=5)
    
    # Save the dataset
    creator.save_dataset(dataset)
    
    # Show some examples
    print("\nSample questions and answers:")
    print("="*50)
    for i, sample in enumerate(dataset, 1):
        print(f"\nSample {i}:")
        print(f"Type: {sample['type']}")
        print(f"Question: {sample['instruction']}")
        print(f"Answer: {sample['response']}")
        print("-"*50)

if __name__ == "__main__":
    main() 
"""
Templatized IFT Dataset Creation with Gemini
============================================

Creates instruction fine-tuning datasets using a structured taxonomy approach.
This version uses templates to generate different types of questions based on
the knowledge taxonomy for more comprehensive and organized datasets.
"""

def main():
    """Main function - demonstrates templatized dataset creation approaches."""
    print("🎯 Templatized IFT Dataset Creation with Gemini")
    
    book_text = load_book()
    if not book_text:
        return
    
    print("\nChoose dataset creation mode:")
    print("1. Comprehensive dataset (all categories, all difficulties)")
    print("2. Specialized dataset (single category and difficulty)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        print("\nCreating comprehensive dataset...")
        dataset = create_comprehensive_dataset(book_text, questions_per_category=2)
        filename = "pride_prejudice_comprehensive_ift.json"
    elif choice == "2":
        print("\nAvailable categories:", categories)
        category = input("Enter category: ").strip()
        
        print("Available difficulties:", difficulties)
        difficulty = input("Enter difficulty: ").strip()
        
        count = int(input("Enter number of questions (default 10): ") or "10")
        
        print(f"\nCreating specialized dataset for {category} ({difficulty})...")
        dataset = create_specialized_dataset(book_text, category, difficulty, count)
        filename = f"pride_prejudice_{category}_{difficulty}_ift.json"
    else:
        print("Invalid choice!")
        return
    
    show_dataset_stats(dataset)
    show_preview(dataset)
    save_dataset(dataset, filename)
    
    print(f"\n🎉 Created {len(dataset)} QA pairs!")

import json
import random
from typing import List, Dict, Any
from google import genai
import os, dotenv

dotenv.load_dotenv()
# Get the API key from environment variables
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Load configuration
with open("4_templates_config.json", 'r') as f:
    config = json.load(f)

# Extract data
templates, characters, themes, events = config["question_templates"], config["characters"], config["themes"], config["events"]
social_aspects, categories, difficulties = config["social_aspects"], config["categories"], config["difficulties"]

def ask_gemini(prompt: str) -> str:
    """Send prompt to Gemini and get response."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return ""

def load_book() -> str:
    """Load Pride and Prejudice text."""
    try:
        import requests
        print("📥 Downloading Pride and Prejudice from Project Gutenberg...")
        text = requests.get("https://www.gutenberg.org/cache/epub/1342/pg1342.txt").text
        print("✅ Text downloaded successfully")
        return text
    except Exception as e:
        print(f"❌ Failed to download text: {e}")
        return ""



def generate_templatized_questions(category: str, difficulty: str, count: int = 5) -> List[str]:
    """Generate questions using templates for a specific category and difficulty."""
    question_templates = templates.get(category, {}).get(difficulty, [])
    if not question_templates:
        return []
    
    questions = []
    for _ in range(count):
        template = random.choice(question_templates)
        
        if category == "character_analysis":
            if "character1" in template and "character2" in template:
                char1 = random.choice(characters["major"])
                char2 = random.choice(characters["major"])
                while char2 == char1:
                    char2 = random.choice(characters["major"])
                question = template.format(character1=char1, character2=char2)
            else:
                character = random.choice(characters["major"] + characters["minor"])
                question = template.format(character=character)
        
        elif category == "themes":
            theme = random.choice(themes)
            if "theme1" in template and "theme2" in template:
                theme1 = random.choice(themes)
                theme2 = random.choice(themes)
                while theme2 == theme1:
                    theme2 = random.choice(themes)
                question = template.format(theme1=theme1, theme2=theme2)
            else:
                question = template.format(theme=theme)
        
        elif category == "plot_events":
            event = random.choice(events)
            question = template.format(event=event)
        
        elif category == "social_context":
            aspect = random.choice(social_aspects)
            question = template.format(aspect=aspect)
        
        else:
            question = template
        
        questions.append(question)
    
    return questions

def get_answer_from_llm(question: str, book_text: str, category: str = "", difficulty: str = "") -> Dict[str, Any]:
    """Create one question-answer pair with metadata."""
    prompt = f"""
    You are an expert on Pride and Prejudice.
    
    Book text: {book_text[:30000]}
    
    Question: {question}
    
    Answer this question accurately and in detail. Provide specific examples from the text when relevant.
    """
    
    answer = ask_gemini(prompt)
    if not answer:
        answer = "Failed to generate answer"
    
    qa_pair = {
        "question": question,
        "answer": answer.strip()
    }
    
    # Add metadata if provided
    if category:
        qa_pair["category"] = category
    if difficulty:
        qa_pair["difficulty"] = difficulty
        qa_pair["type"] = "instruction_following"
    
    return qa_pair



def create_comprehensive_dataset(book_text: str, questions_per_category: int = 3) -> List[Dict[str, Any]]:
    """Create a comprehensive dataset using all categories and difficulty levels."""
    dataset = []
    
    total_questions = len(categories) * len(difficulties) * questions_per_category
    current_question = 0
    
    for category in categories:
        for difficulty in difficulties:
            questions = generate_templatized_questions(category, difficulty, questions_per_category)
            
            for question in questions:
                current_question += 1
                print(f"Progress: {current_question}/{total_questions} - {category} ({difficulty})")
                
                qa_pair = get_answer_from_llm(question, book_text, category, difficulty)
                dataset.append(qa_pair)
    
    return dataset

def create_specialized_dataset(book_text: str, category: str, difficulty: str, count: int = 10) -> List[Dict[str, Any]]:
    """Create a specialized dataset for a specific category and difficulty."""
    dataset = []
    questions = generate_templatized_questions(category, difficulty, count)
    
    for i, question in enumerate(questions, 1):
        print(f"Progress: {i}/{count} - {category} ({difficulty})")
        qa_pair = get_answer_from_llm(question, book_text, category, difficulty)
        dataset.append(qa_pair)
    
    return dataset

def save_dataset(dataset: List[Dict[str, Any]], filename: str):
    """Save dataset to JSON file with metadata."""
    dataset_info = {
        "metadata": {
            "total_samples": len(dataset),
            "categories": list(set(qa.get("category", "") for qa in dataset if qa.get("category"))),
            "difficulties": list(set(qa.get("difficulty", "") for qa in dataset if qa.get("difficulty"))),
            "source": "Pride and Prejudice",
            "generation_method": "templatized_gemini"
        },
        "dataset": dataset
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved to {filename}")
    except Exception as e:
        print(f"❌ Failed to save dataset: {e}")

def show_dataset_stats(dataset: List[Dict[str, Any]]):
    """Show statistics about the generated dataset."""
    print(f"\n📊 Dataset Statistics:")
    print(f"Total samples: {len(dataset)}")
    
    # Category breakdown
    categories = {}
    difficulties = {}
    for qa in dataset:
        cat = qa.get("category", "unknown")
        diff = qa.get("difficulty", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1
    
    print(f"\nCategories:")
    for cat, count in categories.items():
        print(f"  {cat}: {count}")
    
    print(f"\nDifficulties:")
    for diff, count in difficulties.items():
        print(f"  {diff}: {count}")

def show_preview(dataset: List[Dict[str, Any]], num_samples: int = 3):
    """Show preview of the dataset."""
    print(f"\n📖 Preview (first {num_samples} samples):")
    for i, qa in enumerate(dataset[:num_samples], 1):
        print(f"\n{i}. Category: {qa.get('category', 'N/A')} ({qa.get('difficulty', 'N/A')})")
        print(f"   Q: {qa['question']}")
        print(f"   A: {qa['answer'][:150]}...")



if __name__ == "__main__":
    main()

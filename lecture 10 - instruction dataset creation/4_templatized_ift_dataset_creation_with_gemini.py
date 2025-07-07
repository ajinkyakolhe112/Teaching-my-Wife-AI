"""
Templatized IFT Dataset Creation with Gemini
============================================

Creates instruction fine-tuning datasets using a structured taxonomy approach.
This version uses templates to generate different types of questions based on
the knowledge taxonomy for more comprehensive and organized datasets.
"""

import json
import random
from typing import List, Dict, Any
from google import genai

# Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# Configuration file path
TEMPLATES_CONFIG_FILE = "4_templates_config.json"

def load_templates_config():
    """Load templates and configuration from JSON file."""
    try:
        with open(TEMPLATES_CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ Loaded templates from {TEMPLATES_CONFIG_FILE}")
        return config
    except FileNotFoundError:
        print(f"❌ Templates config file not found: {TEMPLATES_CONFIG_FILE}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing templates config: {e}")
        return None

# Load configuration
CONFIG = load_templates_config()
if not CONFIG:
    print("❌ Failed to load configuration. Exiting.")
    exit(1)

# Extract configuration data
QUESTION_TEMPLATES = CONFIG["question_templates"]
CHARACTERS = CONFIG["characters"]
THEMES = CONFIG["themes"]
EVENTS = CONFIG["events"]
SOCIAL_ASPECTS = CONFIG["social_aspects"]
CATEGORIES = CONFIG["categories"]
DIFFICULTIES = CONFIG["difficulties"]

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
        with open("datasets/pride_prejudice.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print("❌ Book file not found: datasets/pride_prejudice.txt")
        return ""

def generate_templatized_questions(category: str, difficulty: str, count: int = 5) -> List[str]:
    """Generate questions using templates for a specific category and difficulty."""
    templates = QUESTION_TEMPLATES.get(category, {}).get(difficulty, [])
    if not templates:
        return []
    
    questions = []
    for _ in range(count):
        template = random.choice(templates)
        
        if category == "character_analysis":
            if "character1" in template and "character2" in template:
                char1 = random.choice(CHARACTERS["major"])
                char2 = random.choice(CHARACTERS["major"])
                while char2 == char1:
                    char2 = random.choice(CHARACTERS["major"])
                question = template.format(character1=char1, character2=char2)
            else:
                character = random.choice(CHARACTERS["major"] + CHARACTERS["minor"])
                question = template.format(character=character)
        
        elif category == "themes":
            theme = random.choice(THEMES)
            if "theme1" in template and "theme2" in template:
                theme1 = random.choice(THEMES)
                theme2 = random.choice(THEMES)
                while theme2 == theme1:
                    theme2 = random.choice(THEMES)
                question = template.format(theme1=theme1, theme2=theme2)
            else:
                question = template.format(theme=theme)
        
        elif category == "plot_events":
            event = random.choice(EVENTS)
            question = template.format(event=event)
        
        elif category == "social_context":
            aspect = random.choice(SOCIAL_ASPECTS)
            question = template.format(aspect=aspect)
        
        else:
            question = template
        
        questions.append(question)
    
    return questions

def create_qa_pair(question: str, book_text: str, category: str, difficulty: str) -> Dict[str, Any]:
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
    
    return {
        "question": question,
        "answer": answer.strip(),
        "category": category,
        "difficulty": difficulty,
        "type": "instruction_following"
    }

def create_comprehensive_dataset(book_text: str, questions_per_category: int = 3) -> List[Dict[str, Any]]:
    """Create a comprehensive dataset using all categories and difficulty levels."""
    dataset = []
    
    total_questions = len(CATEGORIES) * len(DIFFICULTIES) * questions_per_category
    current_question = 0
    
    for category in CATEGORIES:
        for difficulty in DIFFICULTIES:
            questions = generate_templatized_questions(category, difficulty, questions_per_category)
            
            for question in questions:
                current_question += 1
                print(f"Progress: {current_question}/{total_questions} - {category} ({difficulty})")
                
                qa_pair = create_qa_pair(question, book_text, category, difficulty)
                dataset.append(qa_pair)
    
    return dataset

def create_specialized_dataset(book_text: str, category: str, difficulty: str, count: int = 10) -> List[Dict[str, Any]]:
    """Create a specialized dataset for a specific category and difficulty."""
    dataset = []
    questions = generate_templatized_questions(category, difficulty, count)
    
    for i, question in enumerate(questions, 1):
        print(f"Progress: {i}/{count} - {category} ({difficulty})")
        qa_pair = create_qa_pair(question, book_text, category, difficulty)
        dataset.append(qa_pair)
    
    return dataset

def save_dataset(dataset: List[Dict[str, Any]], filename: str):
    """Save dataset to JSON file with metadata."""
    dataset_info = {
        "metadata": {
            "total_samples": len(dataset),
            "categories": list(set(qa["category"] for qa in dataset)),
            "difficulties": list(set(qa["difficulty"] for qa in dataset)),
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

def main():
    """Main function - demonstrates different dataset creation approaches."""
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
        print("\nAvailable categories:", CATEGORIES)
        category = input("Enter category: ").strip()
        
        print("Available difficulties:", DIFFICULTIES)
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

if __name__ == "__main__":
    main()

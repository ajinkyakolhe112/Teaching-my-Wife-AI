"""
Simple IFT Dataset Creation with Gemini
=======================================

Creates instruction fine-tuning datasets using Gemini directly for RAG.
No vector databases or embeddings needed!
"""

def main():
    """Main function - shows the complete flow."""
    print("🎯 Simple IFT Dataset Creation with Gemini")
    
    book_text = load_book()
    
    questions = generate_questions(num_questions=4)
    
    save_questions_to_file(questions)
    
    questions = load_questions_from_file()
    
    dataset = create_qa_pairs_from_llm(questions, book_text)
    
    show_preview(dataset)
    
    save_dataset(dataset)
    
    print(f"\n🎉 Created {len(dataset)} QA pairs!")

import json
from google import genai
import os, dotenv

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

def ask_gemini(prompt):
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

def load_book():
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

def generate_questions(num_questions=10):
    """Generate questions about Pride and Prejudice."""
    prompt = f"""
    Generate {num_questions} questions about Pride and Prejudice.
    Cover: characters, plot, themes, setting.
    Make them varied and interesting.
    Format as numbered list.
    """
    response = ask_gemini(prompt)
    if not response:
        return []
    
    questions = []
    for line in response.split('\n'):
        line = line.strip()
        if line and line[0].isdigit():
            question = line.split('.', 1)[-1].strip()
            if question:
                questions.append(question)
    return questions[:num_questions]

def get_answer_from_llm(question, book_text):
    """Create one question-answer pair."""
    prompt = f"""
    You are an expert on Pride and Prejudice.
    
    Book text: {book_text[:30000]}
    
    Question: {question}
    
    Answer this question accurately and in detail.
    """
    answer = ask_gemini(prompt)
    if not answer:
        return {"question": question, "answer": "Failed to generate answer"}
    
    return {"question": question, "answer": answer.strip()}

def save_questions_to_file(questions, filename="questions.txt"):
    """Save questions to a text file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for i, question in enumerate(questions, 1):
                f.write(f"{i}. {question}\n")
        print(f"✅ Questions saved to {filename}")
    except Exception as e:
        print(f"❌ Failed to save questions: {e}")

def load_questions_from_file(filename="questions.txt"):
    """Load questions from a text file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            questions = []
            for line in f:
                line = line.strip()
                if line and line[0].isdigit():
                    question = line.split('.', 1)[-1].strip()
                    if question:
                        questions.append(question)
            return questions
    except FileNotFoundError:
        print(f"❌ Questions file not found: {filename}")
        return []
    except Exception as e:
        print(f"❌ Failed to load questions: {e}")
        return []

def create_qa_pairs_from_llm(questions, book_text):
    """Create all question-answer pairs."""
    dataset = []
    for i, question in enumerate(questions, 1):
        print(f"Getting answers from LLM. Progress: {i}/{len(questions)}")
        qa_pair = get_answer_from_llm(question, book_text)
        dataset.append(qa_pair)
    return dataset

def save_dataset(dataset, filename="pride_prejudice_ift.json"):
    """Save dataset to JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved to {filename}")
    except Exception as e:
        print(f"❌ Failed to save dataset: {e}")

def show_preview(dataset, num_samples=2):
    """Show preview of the dataset."""
    print(f"\n📖 Preview (first {num_samples} samples):")
    for i, qa in enumerate(dataset[:num_samples], 1):
        print(f"\n{i}. Q: {qa['question']}")
        print(f"   A: {qa['answer'][:100]}...")

if __name__ == "__main__":
    main()

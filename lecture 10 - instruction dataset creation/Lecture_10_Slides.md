# Lecture 10: The Art of the Ask
## Creating High-Impact Custom Instruction Datasets

---

## 🎯 **Learning Objectives**

1. **Explain** why proprietary dataset + commodity model = competitive advantage
2. **Describe** the 5-stage synthetic data pipeline
3. **Implement** templatized IFT dataset creation

---

## 🚀 **Part 1: The Strategic Advantage**

### **The Big Idea**
- **Lecture 7**: Fine-tuned on *Pride and Prejudice* (style)
- **Lecture 8**: Learned IFT (commands)
- **Lecture 10**: Create the **fuel** for IFT

> 💡 **Key Insight**: Pre-training costs millions. Creating specialized data? That's where we innovate.

### **The Great Democratizer of AI => Instruction Fine-tuning Dataset**

#### **Core Thesis:**
**Commodity Model + Proprietary Instruction Fine-tuning Dataset = Competitive LLM with Paid LLMs**

- You don't need to build a new base model
- The real value lies in teaching an existing, powerful model a specialized skill
- Your unique dataset is your defensible advantage

### **The Alpaca Breakthrough (2023)**

| Aspect | Details |
|--------|---------|
| **Base Model** | LLaMA 1 (not comparable to paid LLMs) |
| **Cost** | Just **$600** to fine-tune |
| **Seed Data** | Only **175 human-written** pairs |
| **Generated Data** | **52,000 new instructions** using ChatGPT |
| **Result** | Rivaled GPT-3.5 on instruction-following |

**Revolutionary**: Small investment in data → massive results.

### **AI Development Lifecycle**

1. **🏗️ Pre-training** (Birth - 21): Home schooled student, but hasn't interacted with people or the world. 
2. **🎯 IFT** (Graduate Course): Teach how to be helpful expert
3. **✨ RLHF** (Polish): What should/shouldn't answer

---

## 🛤️ **Part 2: Two Paths to Dataset**

### **Path A: Find It (Fast & Free)**
- ✅ Instant access, no cost
- ❌ General-purpose, no competitive advantage

### **Path B: Synthesize It (Alpaca Way)**
- ✅ Scalable, tailored, proprietary
- ❌ Requires careful process
- 🎯 **Most powerful for most teams**

---

## 🔧 **Part 3: 5-Stage Synthetic Data Pipeline**

### **Stage 1: Gather Raw Knowledge**
```python
import requests
text = requests.get("https://www.gutenberg.org/cache/epub/1342/pg1342.txt").text
```
**Real-world**: Company wiki, support tickets, documentation

### **Stage 2: Generate Seed Instructions**
```python
prompt = """
You are a literary scholar. Generate 20 unique instructions 
based on "Pride and Prejudice." Output as JSON list.
"""
```
**Goal**: Diversity and coverage

### **Stage 3: Generate Grounded Q&A (RAG)**
```python
rag_prompt = """
Using ONLY provided context, answer this instruction.
Answer must be grounded in text provided.

Instruction: "Describe the entail on Longbourn estate"
Context: [Relevant passages...]
"""
```

### **Stage 4: Human Curation**
**Checklist**:
- ✅ Accuracy (100% factual)
- ✅ Helpfulness (teaches useful concept)
- ✅ Clarity & tone
- ✅ Conciseness
- ✅ Completeness

**Golden Rule**: "If you wouldn't want your model to produce this, fix it or discard it."

### **Stage 5: Repeat and Scale**
```python
for instruction in thousands_of_instructions:
    relevant_text = retrieve_from_knowledge_base(instruction)
    answer = generate_with_rag(instruction, relevant_text)
    save_to_dataset(instruction, answer)
```

---

## 💻 **Code Demo: Templatized IFT Creation**

### **Key Features**
- **Structured Taxonomy**: Templates by category & difficulty
- **4 Categories**: Character Analysis, Themes, Plot Events, Social Context
- **3 Difficulties**: Basic, Intermediate, Advanced

### **Configuration Structure**
```json
{
  "question_templates": {
    "character_analysis": {
      "basic": ["Who is {character}?", "What are {character}'s traits?"],
      "intermediate": ["How does {character} change?", "Compare {char1} and {char2}"],
      "advanced": ["Analyze {character}'s psychology", "How does {character} represent themes?"]
    }
  },
  "characters": {
    "major": ["Elizabeth Bennet", "Mr. Darcy", "Jane Bennet"],
    "minor": ["Lydia Bennet", "George Wickham", "Mr. Collins"]
  }
}
```

### **Live Demo**
```python
# Load configuration
config = load_templates_config()

# Generate questions
questions = generate_templatized_questions("character_analysis", "basic", 3)

# Create QA pairs
for question in questions:
    qa_pair = create_qa_pair(question, book_text, category, difficulty)
    dataset.append(qa_pair)

# Save with metadata
save_dataset(dataset, "pride_prejudice_ift.json")
```

### **Sample Output**
```json
{
  "metadata": {
    "total_samples": 24,
    "categories": ["character_analysis", "themes", "plot_events", "social_context"],
    "difficulties": ["basic", "intermediate", "advanced"],
    "source": "Pride and Prejudice"
  },
  "dataset": [
    {
      "question": "Who is Elizabeth Bennet?",
      "answer": "Elizabeth Bennet is the protagonist...",
      "category": "character_analysis",
      "difficulty": "basic"
    }
  ]
}
```

---

## 🎯 **Key Takeaways**

### **Strategic Insights**
1. **Competitive advantage = superior dataset, not bigger model**
2. **Alpaca method = revolutionary low-cost strategy**
3. **5-Stage Pipeline = practical playbook**

### **Templatized Generation Advantages**
- ✅ Better coverage (all aspects covered)
- ✅ Consistent quality (template-based)
- ✅ Metadata rich (category/difficulty info)
- ✅ Customizable (easy to extend)

---

## 🛠️ **Practical Exercise**

### **Assignment: Create Your Own IFT Dataset**
1. Choose a domain (your expertise)
2. Gather raw knowledge (documents)
3. Design question templates
4. Generate 10-20 QA pairs
5. Validate and refine

### **Available Tools**
- `4_templatized_ift_dataset_creation_with_gemini.py`
- `4_templates_config.json`
- `validate_templates.py`
- `simple_demo.py`

---

## 📚 **Resources**

### **Key Papers**
- **Alpaca**: "Self-Instruct: Aligning Language Models with Self-Generated Instructions"
- **Stanford's $600 Fine-tuning**: "Alpaca: A Strong, Replicable Instruction-Following Model"

### **Tools**
- **Hugging Face Hub**: Open-source IFT datasets
- **Google Gemini API**: Synthetic data generation
- **LangChain**: RAG implementations

### **Best Practices**
- Start small (10-20 examples)
- Quality over quantity
- Human curation essential
- Iterate and improve

---

## 🎉 **Conclusion**

You now have the most powerful skill in applied AI: **creating proprietary, high-quality instruction datasets that turn commodity models into specialized experts**.

**Remember**: Future of AI = better data, not bigger models.

---

## 📝 **Discussion Questions**

1. What domain would you create an IFT dataset for?
2. How would you adapt the 5-stage pipeline for your use case?
3. What challenges do you anticipate in human curation?
4. How would you measure dataset quality?

---

*End of Lecture 10* 
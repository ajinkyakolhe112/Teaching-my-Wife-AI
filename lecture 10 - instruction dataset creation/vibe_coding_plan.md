# Harry Potter Synthetic Instruction Dataset Creation Plan

## Overview
This plan outlines a systematic approach to create a high-quality synthetic instruction dataset for Harry Potter books using LLMs and manual prompt engineering techniques.

## Phase 1: Data Preparation & Foundation

### 1.1 Source Material Collection
- **Primary Sources**: All 7 Harry Potter books (text format)
- **Secondary Sources**: 
  - Character wikis and summaries
  - Spell lists and descriptions
  - Location maps and descriptions
  - Timeline of events
  - Magical objects catalog

### 1.2 Knowledge Base Construction
```python
knowledge_base = {
    "characters": {
        "main_characters": ["Harry Potter", "Hermione Granger", "Ron Weasley"],
        "supporting_characters": ["Dumbledore", "Snape", "Voldemort"],
        "houses": ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
    },
    "spells": {
        "combat": ["Expelliarmus", "Stupefy", "Protego"],
        "utility": ["Wingardium Leviosa", "Lumos", "Alohomora"],
        "transfiguration": ["Transfiguration spells"]
    },
    "locations": {
        "hogwarts": ["Great Hall", "Gryffindor Tower", "Dungeons"],
        "magical_world": ["Diagon Alley", "Hogsmeade", "Ministry of Magic"]
    },
    "magical_objects": ["Wand", "Broomstick", "Invisibility Cloak", "Horcruxes"]
}
```

## Phase 2: Prompt Engineering Strategy

### 2.1 Instruction Template Design
Create diverse instruction templates for each category:

```python
instruction_templates = {
    "character_analysis": [
        "Describe the character {character} and their role in the Harry Potter series.",
        "What are {character}'s key personality traits and motivations?",
        "How does {character} evolve throughout the series?",
        "What is {character}'s relationship with Harry Potter?",
        "Explain {character}'s background and family history."
    ],
    
    "plot_explanation": [
        "Explain the events of {event} in the Harry Potter series.",
        "What led to {event} and what were its consequences?",
        "How did {event} impact the main characters?",
        "Describe the timeline of {event}."
    ],
    
    "spell_information": [
        "What is the spell {spell} and what does it do?",
        "How is {spell} used in the Harry Potter series?",
        "What are the effects and limitations of {spell}?",
        "Who invented {spell} and what is its history?"
    ],
    
    "location_description": [
        "Describe {location} and its significance in the Harry Potter world.",
        "What happens at {location} throughout the series?",
        "What are the magical properties of {location}?",
        "How do characters interact with {location}?"
    ],
    
    "magical_object_details": [
        "What is {object} and what are its magical properties?",
        "How is {object} used in the Harry Potter series?",
        "What is the history and significance of {object}?",
        "Who owns or has owned {object}?"
    ]
}
```

### 2.2 Response Quality Control Prompts
```python
quality_prompts = {
    "fact_check": "Verify that the following response about Harry Potter is factually accurate: {response}",
    "completeness": "Ensure this response covers all important aspects of {topic}: {response}",
    "consistency": "Check if this response is consistent with the Harry Potter canon: {response}",
    "detail_level": "Expand this response with more specific details from the books: {response}"
}
```

## Phase 3: LLM-Based Dataset Generation

### 3.1 Multi-Stage Generation Process

#### Stage 1: Content Extraction
```python
def extract_content_segments(book_text):
    """Extract relevant segments from book text"""
    segments = {
        "character_mentions": extract_character_scenes(book_text),
        "spell_usage": extract_spell_instances(book_text),
        "location_descriptions": extract_location_passages(book_text),
        "plot_events": extract_major_events(book_text)
    }
    return segments
```

#### Stage 2: Instruction Generation
```python
def generate_instructions(content_segments):
    """Generate diverse instructions using LLM"""
    instructions = []
    
    for category, segments in content_segments.items():
        for segment in segments:
            prompt = f"""
            Based on this Harry Potter content segment:
            "{segment}"
            
            Generate 3 different types of questions:
            1. A character analysis question
            2. A plot explanation question  
            3. A detail-specific question
            
            Format as JSON:
            {{
                "questions": [
                    {{"instruction": "question text", "type": "category"}}
                ]
            }}
            """
            
            # Use LLM to generate questions
            response = llm_generate(prompt)
            instructions.extend(parse_questions(response))
    
    return instructions
```

#### Stage 3: Response Generation
```python
def generate_responses(instructions, content_segments):
    """Generate high-quality responses using LLM"""
    responses = []
    
    for instruction in instructions:
        # Find relevant content
        relevant_content = find_relevant_content(instruction, content_segments)
        
        prompt = f"""
        Question: {instruction['instruction']}
        
        Relevant content from Harry Potter books:
        {relevant_content}
        
        Provide a comprehensive, accurate answer based on the source material.
        Include specific details, character names, and book references where appropriate.
        Keep the response informative but concise.
        
        Answer:
        """
        
        response = llm_generate(prompt)
        responses.append({
            "instruction": instruction['instruction'],
            "response": response,
            "type": instruction['type'],
            "source_content": relevant_content
        })
    
    return responses
```

### 3.2 Quality Enhancement Pipeline

#### Step 1: Fact Verification
```python
def verify_facts(instruction_response_pairs):
    """Verify factual accuracy using LLM"""
    verified_pairs = []
    
    for pair in instruction_response_pairs:
        verification_prompt = f"""
        Verify the factual accuracy of this Harry Potter response:
        
        Question: {pair['instruction']}
        Answer: {pair['response']}
        
        Check for:
        1. Character name accuracy
        2. Spell name and effect accuracy
        3. Plot event accuracy
        4. Timeline consistency
        
        Return: {{"is_accurate": true/false, "corrections": ["list of corrections"]}}
        """
        
        verification = llm_generate(verification_prompt)
        if verification['is_accurate']:
            verified_pairs.append(pair)
        else:
            # Generate corrected response
            corrected_response = generate_corrected_response(pair, verification['corrections'])
            verified_pairs.append({**pair, 'response': corrected_response})
    
    return verified_pairs
```

#### Step 2: Diversity Enhancement
```python
def enhance_diversity(dataset):
    """Ensure diverse question types and difficulty levels"""
    diversity_metrics = {
        "question_types": count_question_types(dataset),
        "difficulty_levels": assess_difficulty(dataset),
        "character_coverage": check_character_coverage(dataset),
        "book_coverage": check_book_coverage(dataset)
    }
    
    # Generate additional samples to fill gaps
    missing_elements = identify_missing_elements(diversity_metrics)
    additional_samples = generate_targeted_samples(missing_elements)
    
    return dataset + additional_samples
```

## Phase 4: Manual Curation & Refinement

### 4.1 Human Review Process
1. **Accuracy Review**: Verify all facts against source material
2. **Quality Review**: Check for clarity, completeness, and engagement
3. **Diversity Review**: Ensure balanced representation across categories
4. **Difficulty Review**: Maintain appropriate difficulty distribution

### 4.2 Iterative Improvement
```python
def iterative_improvement(dataset, feedback):
    """Improve dataset based on human feedback"""
    improved_dataset = []
    
    for sample in dataset:
        if sample['feedback_score'] < threshold:
            # Regenerate with improved prompts
            improved_sample = regenerate_with_enhanced_prompt(sample)
            improved_dataset.append(improved_sample)
        else:
            improved_dataset.append(sample)
    
    return improved_dataset
```

## Phase 5: Dataset Validation & Testing

### 5.1 Automated Validation
```python
def validate_dataset(dataset):
    """Comprehensive dataset validation"""
    validation_results = {
        "size": len(dataset),
        "diversity_score": calculate_diversity_score(dataset),
        "quality_score": calculate_quality_score(dataset),
        "coverage_score": calculate_coverage_score(dataset),
        "consistency_score": calculate_consistency_score(dataset)
    }
    
    return validation_results
```

### 5.2 Model Testing
```python
def test_on_fine_tuned_model(dataset, model):
    """Test dataset quality by fine-tuning and evaluating"""
    # Split dataset
    train_data, test_data = split_dataset(dataset)
    
    # Fine-tune model
    fine_tuned_model = fine_tune_model(model, train_data)
    
    # Evaluate on test set
    evaluation_metrics = evaluate_model(fine_tuned_model, test_data)
    
    return evaluation_metrics
```

## Phase 6: Implementation Roadmap

### Week 1: Foundation
- [ ] Set up development environment
- [ ] Collect and preprocess source materials
- [ ] Design initial prompt templates
- [ ] Create basic knowledge base

### Week 2: Generation Pipeline
- [ ] Implement content extraction functions
- [ ] Build instruction generation system
- [ ] Create response generation pipeline
- [ ] Set up quality control mechanisms

### Week 3: Quality Enhancement
- [ ] Implement fact verification system
- [ ] Build diversity enhancement tools
- [ ] Create manual review interface
- [ ] Develop iterative improvement process

### Week 4: Validation & Deployment
- [ ] Implement comprehensive validation
- [ ] Test on fine-tuned models
- [ ] Finalize dataset format
- [ ] Document creation process

## Expected Outcomes

### Dataset Specifications
- **Size**: 10,000+ high-quality instruction-response pairs
- **Categories**: 5+ distinct question types
- **Coverage**: All 7 books, major characters, spells, locations
- **Quality**: 95%+ factual accuracy
- **Diversity**: Balanced representation across all categories

### Quality Metrics
- **Factual Accuracy**: >95%
- **Response Completeness**: >90%
- **Question Diversity**: >80% unique question patterns
- **Character Coverage**: All major characters represented
- **Book Coverage**: All 7 books represented

## Tools & Technologies

### LLM Models
- **Primary**: GPT-4 or Claude for generation
- **Secondary**: Local models (LLaMA, Mistral) for testing
- **Verification**: Multiple models for fact-checking

### Development Stack
- **Language**: Python 3.9+
- **Libraries**: Transformers, PyTorch, Pandas, NumPy
- **Tools**: Jupyter Notebooks, VS Code
- **Version Control**: Git with proper branching strategy

### Quality Assurance
- **Automated Testing**: Unit tests for all functions
- **Manual Review**: Human annotation interface
- **Metrics Tracking**: Comprehensive logging and monitoring

This plan provides a systematic approach to creating a high-quality synthetic instruction dataset that can be used for fine-tuning models specifically for Harry Potter domain knowledge. 
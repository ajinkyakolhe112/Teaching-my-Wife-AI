# Templatized IFT Dataset Creation with Gemini

This script creates instruction fine-tuning datasets using a structured taxonomy approach, providing more organized and comprehensive datasets compared to the basic version.

## Key Features

### 🎯 **Structured Taxonomy Approach**
- Uses predefined question templates organized by category and difficulty
- Ensures comprehensive coverage of different aspects of the source material
- Provides consistent question quality and variety

### 📊 **Multiple Categories**
1. **Character Analysis**: Questions about characters, their traits, relationships, and development
2. **Themes**: Questions about major themes like pride, prejudice, love, marriage, social class
3. **Plot Events**: Questions about key events and their significance
4. **Social Context**: Questions about Regency society, customs, and historical background

### 🎓 **Difficulty Levels**
- **Basic**: Simple factual questions
- **Intermediate**: Analytical questions requiring deeper understanding
- **Advanced**: Complex questions involving critical analysis and synthesis

### 🔧 **Flexible Generation Modes**
1. **Comprehensive Mode**: Creates datasets with all categories and difficulty levels
2. **Specialized Mode**: Focuses on specific categories and difficulty levels

## Usage

### Prerequisites
```bash
pip install google-genai
```

### Basic Usage
```bash
python 4_templatized_ift_dataset_creation_with_gemini.py
```

### Validate Configuration
Before running the main script, you can validate the templates configuration:
```bash
python validate_templates.py
```

This will:
- Check the structure of `templates_config.json`
- Validate template variables against available data
- Show a summary of available templates and data
- Display sample questions for each category and difficulty

### Interactive Mode
The script will prompt you to choose:
1. **Comprehensive dataset**: All categories, all difficulties (24 questions total)
2. **Specialized dataset**: Specific category and difficulty

## Dataset Structure

### Output Format
```json
{
  "metadata": {
    "total_samples": 24,
    "categories": ["character_analysis", "themes", "plot_events", "social_context"],
    "difficulties": ["basic", "intermediate", "advanced"],
    "source": "Pride and Prejudice",
    "generation_method": "templatized_gemini"
  },
  "dataset": [
    {
      "question": "Who is Elizabeth Bennet in Pride and Prejudice?",
      "answer": "Elizabeth Bennet is the protagonist...",
      "category": "character_analysis",
      "difficulty": "basic",
      "type": "instruction_following"
    }
  ]
}
```

## Question Templates

### Character Analysis
- **Basic**: "Who is {character}?", "What are the main personality traits of {character}?"
- **Intermediate**: "How does {character} change throughout the novel?", "Compare and contrast {character1} and {character2}"
- **Advanced**: "Analyze the psychological complexity of {character}", "How does {character} represent broader themes?"

### Themes
- **Basic**: "What is the main theme of Pride and Prejudice?", "How is {theme} portrayed in the novel?"
- **Intermediate**: "How does Austen develop the theme of {theme}?", "What is the relationship between {theme1} and {theme2}?"
- **Advanced**: "Analyze how {theme} reflects Regency society", "How does {theme} contribute to the novel's moral philosophy?"

### Plot Events
- **Basic**: "What happens in {event}?", "When does {event} occur in the story?"
- **Intermediate**: "How does {event} affect the plot?", "What are the consequences of {event}?"
- **Advanced**: "Analyze the significance of {event} in the novel's structure", "What does {event} reveal about Regency society?"

### Social Context
- **Basic**: "What was {aspect} like in Regency England?", "How does the novel reflect {aspect} of the time?"
- **Intermediate**: "How does {aspect} influence the characters' decisions?", "What does the novel reveal about {aspect}?"
- **Advanced**: "Analyze how {aspect} shapes the novel's social commentary", "How does {aspect} reflect broader historical changes?"

## Advantages Over Basic Version

### 🎯 **Better Coverage**
- Ensures all major aspects of the source material are covered
- Prevents bias towards certain types of questions
- Provides balanced representation of different difficulty levels

### 📈 **Consistent Quality**
- Template-based generation ensures consistent question structure
- Reduces random or irrelevant questions
- Maintains appropriate difficulty progression

### 🔍 **Metadata Rich**
- Each question includes category and difficulty information
- Enables filtering and analysis of dataset composition
- Useful for training models with specific focus areas

### 🛠️ **Customizable**
- Easy to add new categories or difficulty levels
- Templates can be modified for different source materials
- Supports different question types and formats

## Customization

### Modifying Templates Configuration
All templates and data are now stored in `templates_config.json`. To customize:

1. **Edit the JSON file directly**:
   ```json
   {
     "question_templates": {
       "new_category": {
         "basic": ["Basic question template with {variable}"],
         "intermediate": ["Intermediate question template"],
         "advanced": ["Advanced question template"]
       }
     },
     "new_variables": ["variable1", "variable2", "variable3"]
   }
   ```

2. **Add new categories**: Add to both `question_templates` and `categories` arrays
3. **Add new difficulties**: Add to both template structure and `difficulties` array
4. **Add new data**: Add to appropriate arrays (characters, themes, events, social_aspects)

### Validation
After making changes, run the validator to ensure everything is correct:
```bash
python validate_templates.py
```

### File Structure
```
lecture 10 - instruction dataset creation/
├── 4_templatized_ift_dataset_creation_with_gemini.py  # Main script
├── templates_config.json                              # Templates configuration
├── validate_templates.py                              # Configuration validator
└── README_templatized_ift.md                          # This documentation
```

## Example Output

```
🎯 Templatized IFT Dataset Creation with Gemini

Choose dataset creation mode:
1. Comprehensive dataset (all categories, all difficulties)
2. Specialized dataset (single category and difficulty)

Enter choice (1 or 2): 1

Creating comprehensive dataset...
Progress: 1/24 - character_analysis (basic)
Progress: 2/24 - character_analysis (basic)
...

📊 Dataset Statistics:
Total samples: 24

Categories:
  character_analysis: 6
  themes: 6
  plot_events: 6
  social_context: 6

Difficulties:
  basic: 8
  intermediate: 8
  advanced: 8

📖 Preview (first 3 samples):

1. Category: character_analysis (basic)
   Q: Who is Elizabeth Bennet in Pride and Prejudice?
   A: Elizabeth Bennet is the protagonist and second daughter of the Bennet family...

2. Category: themes (intermediate)
   Q: How does Austen develop the theme of pride?
   A: Austen develops the theme of pride through multiple characters...

3. Category: plot_events (advanced)
   Q: Analyze the significance of Darcy's first proposal in the novel's structure.
   A: Darcy's first proposal serves as a crucial turning point...

✅ Saved to pride_prejudice_comprehensive_ift.json

🎉 Created 24 QA pairs!
```

## Tips for Best Results

1. **Start with Comprehensive Mode**: Get a full overview of your source material
2. **Use Specialized Mode**: Focus on specific areas for targeted training
3. **Review Generated Questions**: Ensure they match your intended difficulty levels
4. **Customize Templates**: Adapt templates for your specific domain or source material
5. **Monitor API Usage**: Be mindful of Gemini API rate limits and costs

## Troubleshooting

### Common Issues
- **API Errors**: Check your Gemini API key and internet connection
- **File Not Found**: Ensure `datasets/pride_prejudice.txt` exists
- **Empty Responses**: May indicate API rate limiting or prompt issues

### Performance Optimization
- Reduce `questions_per_category` for faster generation
- Use specialized mode for targeted dataset creation
- Consider caching responses for repeated runs 
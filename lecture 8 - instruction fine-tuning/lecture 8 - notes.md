# LLM capabilities

Two types of LLMs
1. LLM - Next word predictor
   1. Base Models are only next word predictor
   2. Trained with: "Entire Internet Data"
   3. Easiest to create this data. Uses internet data as is. 
2. LLM - Instruction Follower
   1. Built on top of Base Model
   2. Further Trained with: "Dataset of Instruction and it's answer"
   3. Need to build this dataset

| Feature | Pre-trained Base Model | Instruction Following Model | Chat Model |
|---------|------------|----------------------------|------------|
| Training Data | Raw internet text | Base model + Instruction-response pairs | Base model + Conversational data |
| Primary Function | Next token prediction | Follow specific instructions | Engage in natural conversations |
| Example | GPT-3 base | GPT-3 Instruct | GPT-3.5 Turbo |
| Use Case | Text completion | Task-specific instructions | Interactive conversations |
| Training Complexity | Moderate | High (needs instruction dataset) | Highest (needs conversation dataset) |
| Output Style | Raw, unfiltered | Structured, task-focused | Natural, conversational |
| Safety Measures | Minimal | Moderate | Extensive |
| Fine-tuning Required | No | Yes | Yes |

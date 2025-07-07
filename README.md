# 0 to LLM Workshop: Complete 12-Lecture Curriculum

This is the complete curriculum for the "Teaching my Wife AI - 0 to ChatGPT" workshop, a comprehensive 12-lecture course designed to take Shreya from zero knowledge to building and deploying custom Large Language Model (LLM) applications.

## 🎯 Workshop Overview

This 12-lecture workshop provides a complete journey from LLM fundamentals to advanced application development. Through structured progression, hands-on implementation, and practical focus, you'll gain both theoretical understanding and practical skills needed to work effectively with Large Language Models in real-world scenarios.

### Core Mental Model: Human Development Analogy

The workshop is built around three fundamental AI training methods explained through human development:

1. **Pre-training**: Raising a child (Birth to Age 21) - Massive time investment, very high cost
2. **RAG**: Graduate with Library Access - Medium time investment, moderate cost, always needs library access  
3. **Fine-tuning**: Graduate Course - Short time investment (2 years), lower cost, learns information so well it doesn't need to look it up

## 📚 Complete Lecture-by-Lecture Breakdown

### Phase 1: Foundation (Lectures 1-3)

#### Lecture 1: Introduction to LLMs
- **Content**: Basic concepts and terminology of Large Language Models
- **Key Concepts**: Understanding what LLMs are and their capabilities, basic terminology and architecture overview
- **Practical Component**: Using Ollama for local LLMs
- **Skills Built**: Basic LLM interaction, environment setup
- **Files**: `lecture 1 - overview of LLMs/using ollama for local LLMs.md`

#### Lecture 2: Intro to Deep Learning on Kaggle
- **Content**: Neural network fundamentals and practical implementation
- **Key Concepts**: Single neuron implementation, multi-layer neural networks, 3-layer architecture with ReLU activation
- **Practical Component**: California Housing dataset regression using TensorFlow/Keras
- **Skills Built**: Understanding neural network architecture, training process, validation
- **Technical Stack**: TensorFlow/Keras, NumPy, Matplotlib
- **Files**: `lecture 2 - intro to DL with kaggle/intro_to_dl_kaggle.ipynb`

#### Lecture 3: Llama Research Paper Reading & Simple LLMs
- **Content**: Practical interaction with LLMs using Hugging Face transformers
- **Key Concepts**: Loading and using pre-trained models, tokenization and generation process, chat formatting
- **Practical Component**: Chat interface with Llama models, simple chat script implementation, Ollama integration
- **Skills Built**: Model loading, text generation, chat interfaces
- **Technical Stack**: Hugging Face Transformers, Ollama SDK
- **Files**: `lecture 3 - llama research paper reading/simple_chat.py`, `ollama.ipynb`, `simple_llms.ipynb`

### Phase 2: Practical Skills (Lectures 4-6)

#### Lecture 4: Simple Sentiment Analysis with Hugging Face Transformers
- **Content**: Using pre-trained models for sentiment analysis
- **Key Concepts**: Hugging Face pipeline abstraction, model components, DistilBERT architecture
- **Practical Component**: Sentiment analysis pipeline, model inspection and understanding
- **Skills Built**: Using pre-trained models, understanding model components
- **Technical Stack**: Hugging Face Transformers, DistilBERT
- **Files**: `lecture 4 - simple sentiment analysis with huggingface transformers/sentiment_analysis_with_transformers.ipynb`

#### Lecture 5: Simple Sentiment Analysis with PyTorch
- **Content**: Building sentiment analysis from scratch using PyTorch
- **Key Concepts**: Custom tokenizer implementation, simple neural network architecture, data preprocessing
- **Practical Component**: SimpleTokenizer class, SimpleSentimentModel with embedding and linear layers
- **Skills Built**: Building models from scratch, custom preprocessing
- **Technical Stack**: PyTorch, Custom architectures, IMDB dataset
- **Files**: `lecture 5 - simple sentiment analysis with pytorch/simple_sentiment_model.py`, `train_model.py`

#### Lecture 6: Kaggle Competition for Sentiment Analysis
- **Content**: Real-world disaster tweet classification competition
- **Key Concepts**: Real-world dataset handling, advanced model architectures, competition-style evaluation
- **Practical Component**: Disaster tweet classification, advanced model training, competition submission preparation
- **Skills Built**: Production-ready model development, competition participation
- **Technical Stack**: KerasNLP, Advanced preprocessing techniques
- **Files**: `lecture 6 - kaggle competition sentiment analysis/disaster-tweet-sb.ipynb`, `disaster-tweet-code_assistant.ipynb`

### Phase 3: Customization (Lectures 7-8)

#### Lecture 7: LLM Fine-tuning
- **Content**: Understanding and implementing fine-tuning using Pride and Prejudice
- **Key Concepts**: Fine-tuning vs pre-training vs RAG comparison, human development analogy, practical fine-tuning implementation
- **Practical Component**: Fine-tuning on Pride and Prejudice text, class-based and function-based implementations
- **Skills Built**: Fine-tuning process, domain adaptation, training monitoring
- **Technical Stack**: Hugging Face Transformers, WandB, Custom datasets
- **Files**: `lecture 7 - finetuning llms/pride_prejudice_finetune_class.py`, `pride_prejudice_finetune_functions.py`, `analogy_code.py`

#### Lecture 8: Instruction Fine-tuning
- **Content**: Creating instruction-following models
- **Key Concepts**: Base models vs instruction-following models vs chat models, instruction dataset requirements
- **Practical Component**: Multiple fine-tuning approaches (Alpaca, FLAN, Code Alpaca), model testing and comparison
- **Skills Built**: Instruction fine-tuning, model comparison, dataset understanding
- **Technical Stack**: Multiple fine-tuning approaches, Model evaluation
- **Files**: `lecture 8 - instruction fine-tuning/alpaca_finetune.py`, `flan_finetune.py`, `code_alpaca_finetune.py`

### Phase 4: Application & Advanced Techniques (Lectures 9-11)

#### Lecture 9: Running LLM Applications
- **Content**: Building interactive applications with Gradio
- **Key Concepts**: Web interface creation, multi-modal applications, real-time processing
- **Practical Component**: Chatbot in browser, musical tone generation, speech-to-text conversion, live image classification
- **Skills Built**: Application deployment, multi-modal interfaces, user experience
- **Technical Stack**: Gradio, Audio processing libraries, Image classification models
- **Files**: `lecture 9 - running llm applications/2 - chatbot model in browser.ipynb`, `2 - simple musical application.ipynb`, `3 - speech to text in browser.ipynb`, `4 - live image classification.ipynb`

#### Lecture 10: Creating Custom Instruction Datasets
- **Content**: The art of creating high-impact custom instruction datasets
- **Key Concepts**: Strategic advantage of proprietary datasets, essential qualities (Diversity, Factual Grounding, Quality), Alpaca breakthrough methodology
- **Practical Component**: 5-stage synthetic data pipeline, RAG-based dataset generation, human curation and refinement
- **Skills Built**: Dataset creation, RAG implementation, quality assurance
- **Technical Stack**: RAG systems, LangChain, Custom generation pipelines
- **Files**: `lecture 10 - instruction dataset creation/langchain_rag.py`, `lecture_10_ift_dataset_creation.py`, `ultra_minimal_ift.py`

#### Lecture 11: Parameter Efficient Fine-tuning (PEFT)
- **Content**: Advanced fine-tuning techniques using LoRA
- **Key Concepts**: Parameter-Efficient Fine-tuning (PEFT) concepts, LoRA methodology, computational cost reduction strategies
- **Practical Component**: LoRA configuration and implementation, fine-tuning on custom instruction datasets, web interface for model interaction
- **Skills Built**: Advanced fine-tuning, PEFT techniques, model optimization
- **Technical Stack**: PEFT library, LoRA, Hugging Face Transformers
- **Files**: `lecture 11 - parameter efficient fine tuning/peft_fine_tuning.py`, `fine_tune_pride_prejudice.py`, `web_interface.py`

#### Lecture 12: End-to-End LLM App Development
- **Content**: Building complete LLM applications (Directory exists but content not yet developed)
- **Key Concepts**: Complete application architecture, production deployment considerations, performance optimization
- **Status**: Framework established, content development pending
- **Files**: `lecture 12 - end to end LLM App Dev/` (directory structure ready)

## 🛠️ Technical Stack Covered

#### Core Frameworks
- **TensorFlow/Keras**: Neural network implementation
- **PyTorch**: Custom model development
- **Hugging Face Transformers**: Pre-trained model usage
- **Gradio**: Web interface creation

#### Specialized Libraries
- **PEFT**: Parameter-efficient fine-tuning
- **LangChain**: RAG implementation
- **Ollama**: Local LLM deployment
- **WandB**: Experiment tracking

#### Models & Architectures
- **DistilBERT**: Sentiment analysis
- **Llama-2**: Chat and instruction models
- **TinyLlama**: Lightweight models
- **Custom architectures**: Built from scratch

#### Datasets
- **IMDB**: Sentiment analysis
- **California Housing**: Regression
- **Disaster Tweets**: Real-world classification
- **Pride and Prejudice**: Custom domain adaptation

---

## 🎓 Learning Outcomes

By the end of this workshop, you will be able to:

### Technical Skills
- **Understand LLM Fundamentals**: From basic concepts to advanced architectures
- **Build Custom Models**: Fine-tune models for specific domains and tasks
- **Create Specialized Datasets**: Generate high-quality instruction datasets
- **Deploy Applications**: Build and deploy LLM-powered applications
- **Solve Real-world Problems**: Apply LLM techniques to practical challenges

### Conceptual Understanding
- **AI Development Lifecycle**: Pre-training → Fine-tuning → RAG → RLHF
- **Human Development Analogy**: Making complex AI concepts accessible
- **Strategic Thinking**: Understanding competitive advantages in AI
- **Quality Assurance**: Dataset creation and model evaluation


---

**Ready to start your LLM journey?** Begin with [Lecture 1](lecture%201%20-%20overview%20of%20LLMs/) and follow the progressive curriculum to build your expertise in Large Language Models!
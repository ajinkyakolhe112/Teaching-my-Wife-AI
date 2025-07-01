# 7-Week Intensive LLM Workshop

## Overview/Introduction

Welcome to the 7-Week Intensive LLM Workshop! This workshop is designed to take you from the foundational concepts of Large Language Models (LLMs) to practical applications like Retrieval Augmented Generation (RAG) and sentiment analysis. You'll learn about the core mechanisms that power LLMs, how to interact with them effectively, and how to build simple applications using them.

This is a hands-on workshop featuring:
*   **Weekly Lectures:** Covering theoretical concepts and practical code.
*   **Lab Sessions:** Providing guided, hands-on experience with the tools and techniques discussed.
*   **Assignments:** Challenging you to apply your learning and explore further.

**Target Audience:**
This workshop is aimed at developers, students, NLP enthusiasts, and anyone looking to gain a solid understanding of Large Language Models and learn how to use them effectively.

## Prerequisites (General)

*   **Basic Python Programming Knowledge:** You should be comfortable reading and writing simple Python scripts.
*   **Familiarity with Command-Line Interface (CLI):** Basic navigation and running commands in a terminal.
*   **(Optional but helpful) Basic Understanding of Machine Learning Concepts:** Familiarity with terms like training, datasets, and evaluation will be beneficial but is not strictly required.

## Tools and Technologies Used

Throughout this workshop, we will be using a variety of tools and Python libraries:

*   **Python 3.x**
*   **Ollama:** For running Large Language Models locally.
*   **Hugging Face Libraries:**
    *   `transformers`: For accessing and using pre-trained models.
    *   `datasets`: For loading and working with common datasets.
    *   `sentence-transformers`: For generating text embeddings.
*   **PyTorch:** An open-source machine learning framework used for building and training models.
*   **LangChain:** A framework for developing applications powered by language models (used in Week 6 for RAG).
*   **Jupyter Notebooks:** For interactive coding and lecture notes.
*   **Standard Data Science Libraries:**
    *   `numpy`: For numerical operations.
    *   `matplotlib`: For plotting.
    *   `scikit-learn`: For machine learning utilities.
*   **Weights & Biases (`wandb`):** For experiment tracking in some PyTorch exercises.
*   **FAISS:** For efficient similarity search in vector databases (used in RAG).

## Workshop Structure (6 Weeks)

Here is a week-by-week breakdown of the workshop content:

---

### Week 1: Introduction to LLMs and Local Setup

*   **Description:** This week introduces the fundamental concepts of Large Language Models, their capabilities, limitations, and ethical considerations. You'll set up your local environment using Ollama to run LLMs on your own machine.
*   **Key Learning Objectives:**
    *   Understand what LLMs are, their history, and common architectures.
    *   Learn to install and use Ollama to download and run local LLMs.
    *   Gain initial experience interacting with LLMs via CLI and Python SDK.
*   **Materials:**
    *   `[Week 1 Lecture](./week1/lecture/)`
    *   `[Week 1 Lab](./week1/lab/)`
    *   `[Week 1 Assignment](./week1/assignment/)`

---

### Week 2: Core Concepts - Embeddings, Attention, and Transformers

*   **Description:** Dive into the foundational mechanisms of LLMs. This week covers word embeddings, the attention mechanism, and the Transformer architecture that underpins most modern LLMs.
*   **Key Learning Objectives:**
    *   Understand what word embeddings are and how they represent meaning.
    *   Grasp the intuition and basic mechanics of the attention mechanism.
    *   Learn about the key components of the Transformer architecture (Encoder-Decoder, Self-Attention).
*   **Materials:**
    *   `[Week 2 Lecture](./week2/lecture/)`
    *   `[Week 2 Lab](./week2/lab/)`
    *   `[Week 2 Assignment](./week2/assignment/)`

---

### Week 3: Interacting with LLMs and Prompt Engineering

*   **Description:** Learn practical ways to interact with LLMs using libraries like Hugging Face `transformers` and the Ollama SDK. This week also introduces key insights from influential research (e.g., LLaMA paper) and foundational prompt engineering techniques.
*   **Key Learning Objectives:**
    *   Interact with pre-trained models from Hugging Face Hub using the `transformers` library.
    *   Understand key insights from the LLaMA paper relevant to practical usage.
    *   Develop foundational skills in prompt engineering (zero-shot, few-shot, role prompting).
*   **Materials:**
    *   `[Week 3 Lecture](./week3/lecture/)`
    *   `[Week 3 Lab](./week3/lab/)`
    *   `[Week 3 Assignment](./week3/assignment/)`

---

### Week 4: Fine-tuning LLMs for Specific Tasks (Conceptual)

*   **Description:** This week provides a conceptual overview of fine-tuning Large Language Models. You'll learn why fine-tuning is important, different methods like LoRA and QLoRA, and the general process involved, without necessarily performing full fine-tuning locally.
*   **Key Learning Objectives:**
    *   Understand the concept of fine-tuning and its benefits.
    *   Learn about different fine-tuning techniques, including full fine-tuning and Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA and QLoRA.
    *   Get an overview of the tools, datasets, and evaluation strategies involved in fine-tuning.
*   **Materials:**
    *   `[Week 4 Lecture](./week4/lecture/)`
    *   `[Week 4 Lab](./week4/lab/)`
    *   `[Week 4 Assignment](./week4/assignment/)`

---

### Week 5: Sentiment Analysis

*   **Description:** Explore sentiment analysis using two approaches: leveraging Hugging Face pipelines for quick and effective results with pre-trained models, and building a simple sentiment analysis model from scratch using PyTorch to understand the underlying mechanics.
*   **Key Learning Objectives:**
    *   Understand how LLMs can be used for sentiment analysis.
    *   Implement sentiment analysis using Hugging Face Pipelines.
    *   Gain conceptual understanding of building a sentiment analysis model with PyTorch (tokenizer, model architecture, training loop).
    *   Compare and contrast these different approaches.
*   **Materials:**
    *   `[Week 5 Lecture](./week5/lecture/)`
    *   `[Week 5 Lab](./week5/lab/)`
    *   `[Week 5 Assignment](./week5/assignment/)`

---

### Week 6: Retrieval Augmented Generation (RAG)

*   **Description:** Learn about Retrieval Augmented Generation (RAG), a powerful technique to connect LLMs to external knowledge sources. This week covers the components of a RAG system (vector databases, retrievers) and how it helps reduce hallucinations and incorporate custom data.
*   **Key Learning Objectives:**
    *   Understand the concept of RAG and its benefits (addressing outdated knowledge, using custom data).
    *   Learn the components of a basic RAG system (document loading, chunking, embeddings, vector stores, retriever, LLM).
    *   Build a simple RAG pipeline using LangChain, a local LLM (Ollama), and a local vector store (FAISS).
*   **Materials:**
    *   `[Week 6 Lecture](./week6/lecture/)`
    *   `[Week 6 Lab](./week6/lab/)`
    *   `[Week 6 Assignment](./week6/assignment/)`

### Week 7: Building Simple LLM Applications: (Remaining to create it's Lecture, Lab & Assignment)
---

## Setup Instructions

1.  **Python Environment:**
    It is highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python3 -m venv llm_workshop_env
    source llm_workshop_env/bin/activate  # On Windows use `llm_workshop_env\\Scripts\\activate`
    ```

2.  **Ollama Installation:**
    Ollama is used for running LLMs locally. Please refer to the detailed installation guide provided in Week 1:
    *   `[Ollama Installation Guide](./week1/lecture/week1_lecture2_ollama_local_llms.md#installation-guide)`
    After installation, ensure Ollama is running and you have pulled necessary models as specified in the weekly materials (e.g., `ollama pull phi3:mini`).

3.  **Python Packages:**
    Each week's lab and assignment materials may specify particular Python packages that need to be installed. Generally, you will use `pip install <package_name>`. Common packages are listed under "Tools and Technologies Used." It's good practice to install them within your virtual environment.
    A general list of packages used across the workshop is:
    ```bash
    pip install torch transformers datasets sentence-transformers langchain langchain_community faiss-cpu ollama jupyter numpy matplotlib scikit-learn wandb
    ```
    Refer to individual lab/assignment files for any specific versions or additional packages if you encounter issues.

## How to Use This Repository

1.  **Clone the Repository (if you haven't already):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Activate Your Virtual Environment.**
3.  **Navigate to the Respective Week's Directory:**
    Each week's content is organized into `lecture`, `lab`, and `assignment` subdirectories.
    ```bash
    cd week1 
    # or week2, week3, etc.
    ```
4.  **Follow the Materials:**
    *   Start with the materials in the `lecture` folder. These are often Jupyter Notebooks (`.ipynb`) or Markdown files (`.md`).
    *   Proceed to the `lab` folder for hands-on exercises.
    *   Finally, complete the tasks in the `assignment` folder.

## License

This content is provided for educational purposes. Please refer to the licenses of individual tools and libraries used (Ollama, Hugging Face, PyTorch, LangChain, etc.) for their specific terms of use.

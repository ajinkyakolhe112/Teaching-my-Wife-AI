# 7-Week LLM Workshop Structure

## Week 1: Introduction to LLMs and Local Setup

**Learning Objectives:**
* Understand the fundamental concepts of Large Language Models (LLMs).
* Set up a local environment for running LLMs using Ollama.
* Gain initial experience interacting with a local LLM.

**Lecture Topics:**
*   **Overview of LLMs:** What are LLMs, their capabilities, and common use cases. (Leverages existing Lecture 1 ideas but needs content creation).
    *   Proposed format: Markdown document.
*   **Introduction to Ollama:** What is Ollama, why use it for local LLMs, and how to install it. (Leverages existing Lecture 1 ideas but needs content creation).
    *   Proposed format: Markdown document with installation steps.
*   **Running Your First Local LLM:** Step-by-step guide to downloading and running a model (e.g., Llama 2, Mistral) using Ollama. (Leverages existing Lecture 1 ideas but needs content creation).
    *   Proposed format: Jupyter Notebook.

**Lab Session:**
*   **Ollama Setup and First Interaction:** Guided session to ensure Ollama is installed correctly and participants can run a pre-selected LLM. Participants will try out a few basic prompts.
    *   Proposed format: Guided Jupyter Notebook.

**Assignment:**
*   **Explore Ollama Models:** Research and try running at least two different LLMs available through Ollama. Document your experience and any interesting observations.
    *   Proposed format: Q&A (short written report).

## Week 2: Core Concepts - Embeddings, Attention, Transformers

**Learning Objectives:**
* Understand the foundational concepts of embeddings and their role in LLMs.
* Grasp the intuition behind the attention mechanism.
* Learn about the Transformer architecture and its significance.

**Lecture Topics:**
*   **Embeddings:** What are word embeddings, sentence embeddings, and how they represent text numerically. (New Lecture 2, to be developed).
    *   Proposed format: Markdown document with diagrams.
*   **Attention Mechanism:** Conceptual overview of attention, how it helps models focus on relevant parts of input. (New Lecture 2, to be developed).
    *   Proposed format: Markdown document with simplified examples.
*   **Transformers Architecture:** High-level overview of the encoder-decoder structure, self-attention, and feed-forward networks. (New Lecture 2, to be developed).
    *   Proposed format: Markdown document with diagrams.

**Lab Session:**
*   **Visualizing Embeddings (Conceptual):** Explore pre-computed embeddings and visualize their relationships (e.g., using t-SNE in a Jupyter Notebook). This would be more about understanding than building.
    *   Proposed format: Guided Jupyter Notebook.

**Assignment:**
*   **Transformer Explanation:** In your own words, explain the role of self-attention in the Transformer model.
    *   Proposed format: Q&A (short written explanation).

## Week 3: Interacting with LLMs and Prompt Engineering

**Learning Objectives:**
* Learn how to interact with LLMs using the Hugging Face `transformers` library.
* Understand key insights from the LLaMA paper relevant to practical usage.
* Develop foundational skills in prompt engineering.

**Lecture Topics:**
*   **Hugging Face `transformers` Library:** Introduction to the library, loading pre-trained models, tokenizers, and basic inference. (Existing Lecture 3, to be enhanced).
    *   Proposed format: Jupyter Notebook.
*   **LLaMA Paper Insights:** Key takeaways from the LLaMA paper focusing on model scale, training data, and performance. (Existing Lecture 3, to be enhanced).
    *   Proposed format: Markdown document summarizing key points.
*   **Prompt Engineering Fundamentals:** Basic prompting techniques (zero-shot, few-shot), instruction tuning, and common pitfalls. (Existing Lecture 3, to be enhanced).
    *   Proposed format: Markdown document with examples.

**Lab Session:**
*   **Hugging Face Inference and Basic Prompting:** Hands-on practice with loading a model from Hugging Face and experimenting with different prompts to achieve desired outputs for simple tasks.
    *   Proposed format: Guided Jupyter Notebook.

**Assignment:**
*   **Prompt Engineering Challenge:** Given a specific task (e.g., summarization, question answering), design 3 different prompts and compare their outputs from a chosen LLM.
    *   Proposed format: Coding problem (Python script using Hugging Face) and short analysis.

## Week 4: Fine-tuning LLMs (Conceptual)

**Learning Objectives:**
* Understand the concept of fine-tuning and why it's important.
* Learn about different fine-tuning techniques like PEFT, LoRA, and QLoRA at a conceptual level.
* Get an overview of tools and platforms used for fine-tuning.

**Lecture Topics:**
*   **What is Fine-tuning?:** Why fine-tune, when to fine-tune, and the difference between pre-training and fine-tuning. (New Lecture, Week 4, to be developed).
    *   Proposed format: Markdown document.
*   **Parameter Efficient Fine-Tuning (PEFT):** Introduction to PEFT and its advantages. (New Lecture, Week 4, to be developed).
    *   Proposed format: Markdown document.
*   **LoRA and QLoRA:** Conceptual explanation of Low-Rank Adaptation and Quantized LoRA. (New Lecture, Week 4, to be developed).
    *   Proposed format: Markdown document with diagrams.
*   **Tools for Fine-tuning:** Overview of libraries like Hugging Face `trl`, Axolotl, and platforms. (New Lecture, Week 4, to be developed).
    *   Proposed format: Markdown document.

**Lab Session:**
*   **Exploring Fine-tuning Configurations (Conceptual):** Review examples of fine-tuning scripts and configurations (e.g., from Hugging Face `trl` examples). Discuss how parameters would be set for a hypothetical task. No actual training due to resource constraints, focus on understanding the setup.
    *   Proposed format: Guided discussion and review of code examples (Jupyter Notebook or Python scripts).

**Assignment:**
*   **Fine-tuning Use Case Proposal:** Identify a specific task or dataset where fine-tuning an LLM could be beneficial. Describe the problem, why fine-tuning is appropriate, and which conceptual technique (LoRA, QLoRA) might be suitable.
    *   Proposed format: Research task (written proposal).

## Week 5: Sentiment Analysis with LLMs

**Learning Objectives:**
* Understand how to use LLMs for sentiment analysis.
* Implement sentiment analysis using Hugging Face Pipelines.
* (Optional Stretch Goal) Understand the basics of building a sentiment analysis model with PyTorch.

**Lecture Topics:**
*   **Sentiment Analysis with LLMs:** How LLMs can be prompted or fine-tuned for sentiment classification. (Existing Lectures 4 & 5, to be structured).
    *   Proposed format: Markdown document.
*   **Hugging Face Pipelines for Sentiment Analysis:** Using the `sentiment-analysis` pipeline for quick and easy sentiment classification. (Existing Lectures 4 & 5, to be structured).
    *   Proposed format: Jupyter Notebook.
*   **Sentiment Analysis from Scratch with PyTorch (Conceptual Overview):** Briefly cover dataset preparation, model architecture (e.g., using a pre-trained encoder), and training loop. (Existing Lectures 4 & 5, to be structured - focus on concepts unless time permits a deeper dive).
    *   Proposed format: Jupyter Notebook or Markdown with code snippets.

**Lab Session:**
*   **Practical Sentiment Analysis:** Use Hugging Face Pipelines to perform sentiment analysis on various text samples. Experiment with different models available through the pipeline.
    *   Proposed format: Guided Jupyter Notebook.

**Assignment:**
*   **Movie Review Sentiment Analyzer:** Given a small dataset of movie reviews, use Hugging Face Pipelines to classify their sentiment. Report accuracy or provide examples of correct and incorrect classifications.
    *   Proposed format: Coding problem (Python script using Hugging Face).

## Week 6: Retrieval Augmented Generation (RAG)

**Learning Objectives:**
* Understand the concept of Retrieval Augmented Generation (RAG) and its benefits.
* Learn the components of a basic RAG system (vector databases, retrievers).
* Explore how RAG can reduce hallucinations and provide up-to-date information.

**Lecture Topics:**
*   **Introduction to RAG:** What is RAG, why it's needed (limitations of LLM knowledge, hallucinations), and common use cases. (New Lecture, Week 6, to be developed).
    *   Proposed format: Markdown document.
*   **Components of a RAG System:**
    *   **Vector Databases:** Overview of ChromaDB, FAISS, etc. (New Lecture, Week 6, to be developed).
        *   Proposed format: Markdown document.
    *   **Retrievers:** How documents are retrieved based on query similarity. (New Lecture, Week 6, to be developed).
        *   Proposed format: Markdown document.
    *   **LLM as Generator:** How the LLM uses retrieved context to generate answers. (New Lecture, Week 6, to be developed).
        *   Proposed format: Markdown document.
*   **Building a Simple RAG (Conceptual):** High-level steps to implement a RAG system (document loading, chunking, embedding, indexing, retrieval, generation). (New Lecture, Week 6, to be developed).
    *   Proposed format: Jupyter Notebook or Markdown with pseudocode.

**Lab Session:**
*   **Mini-RAG with a Small Document Set:** Implement a very basic RAG system using a few local text files, a simple vector store (e.g., FAISS or ChromaDB in-memory), and an LLM (via Ollama or Hugging Face) to answer questions based on the documents.
    *   Proposed format: Guided Jupyter Notebook.

**Assignment:**
*   **RAG Design for a Q&A Bot:** Design a RAG system for a Q&A bot for a specific domain (e.g., a company's FAQ, a specific textbook). Describe the document sources, choice of vector DB (conceptual), and how you would evaluate its performance.
    *   Proposed format: Research task (written design document).

## Week 7: Building Simple LLM Applications

**Learning Objectives:**
* Learn how to build simple web interfaces for LLM applications.
* Get introduced to Gradio for rapid prototyping.
* Get introduced to Streamlit as an alternative for building data apps.

**Lecture Topics:**
*   **Why Build UIs for LLMs?:** Accessibility, interactivity, and showcasing LLM capabilities. (New Lecture, Week 7, to be developed).
    *   Proposed format: Markdown document.
*   **Introduction to Gradio:** Core concepts, creating simple interfaces (text input/output), and integrating with LLM backends. (New Lecture, Week 7, to be developed).
    *   Proposed format: Jupyter Notebook or Python scripts.
*   **Introduction to Streamlit:** Core concepts, building interactive widgets, and deploying simple apps. (New Lecture, Week 7, to be developed).
    *   Proposed format: Python scripts.
*   **Example Application: Simple Chatbot:** Walkthrough of building a basic chatbot interface using Gradio or Streamlit that interacts with a local LLM. (New Lecture, Week 7, to be developed).
    *   Proposed format: Python scripts with explanations.

**Lab Session:**
*   **Build Your Own Gradio/Streamlit App:** Participants will build a simple UI (e.g., a text summarizer or a Q&A interface) for an LLM they've worked with previously, using either Gradio or Streamlit.
    *   Proposed format: Python script with instructions, choice of Gradio or Streamlit.

**Assignment:**
*   **Enhance Your LLM App:** Add a new feature to the application built during the lab session (e.g., history, ability to choose different LLMs, more sophisticated UI elements).
    *   Proposed format: Coding problem (Python script).



This structure provides a comprehensive plan. Content for new lectures will need to be created, and existing material will be adapted and enhanced as noted.

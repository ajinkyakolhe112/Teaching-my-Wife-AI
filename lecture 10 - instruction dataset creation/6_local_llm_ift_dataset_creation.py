"""
Pride and Prejudice Instruction Fine-Tuning Dataset Creator

This module provides a complete pipeline for creating instruction fine-tuning datasets
from Jane Austen's "Pride and Prejudice" using Gemini Pro and LlamaIndex RAG.
"""

import os
import requests
from google import genai
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

class GeminiProClient:
    """Client for interacting with Google's Gemini Pro API."""

    def __init__(self, api_key="AIzaSyCEsL5NELX_0ZyG5AHnJCZgkpczCtXLG8Q"):
        self.client = genai.Client(api_key=api_key)

    def generate_response(self, prompt):
        """Generate a response using Gemini Pro."""
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"Gemini Pro error: {e}")
            return None


class PridePrejudiceRAG:
    """RAG system for querying Pride and Prejudice knowledge base using LangChain."""

    def __init__(self):
        self.qa_chain = None
        self._setup_rag()

    def _setup_rag(self):
        """Initialize the RAG system with Pride and Prejudice text using LangChain."""
        try:
            # Step 1: Document Loading - Get text from Project Gutenberg
            print("📥 Downloading Pride and Prejudice from Project Gutenberg...")
            text = requests.get("https://www.gutenberg.org/cache/epub/1342/pg1342.txt").text
            documents = [Document(page_content=text, metadata={"source": "project_gutenberg"})]
            print("✅ Text downloaded successfully")

            # Step 2: Text Chunking - Split large documents into smaller, manageable pieces
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)

            # Step 3: Embedding Generation - Convert text chunks into numerical vectors
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

            # Step 4: Vector Storage - Store embeddings in a searchable vector database
            vectorstore = FAISS.from_documents(chunks, embeddings)

            # Step 5: LLM Setup - Initialize the language model for text generation
            llm = HuggingFacePipeline(
                pipeline=pipeline("text-generation", model="microsoft/DialoGPT-medium", max_length=512)
            )

            # Step 6: RAG Chain Creation - Combine retrieval and generation into a single chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff", 
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
            )

            print("✅ RAG system initialized successfully with LangChain")

        except Exception as e:
            print(f"❌ RAG setup failed: {e}")

    def query(self, question):
        """Query the RAG system with a question."""
        if not self.qa_chain:
            return "RAG system not initialized"
        
        try:
            response = self.qa_chain.run(question)
            return response
        except Exception as e:
            return f"Error: {e}"


class TaxonomyTemplate:
    """Template system for generating structured taxonomies."""
    
    def __init__(self):
        self.templates = {
            "literary_analysis": {
                "categories": [
                    "Characters and Relationships",
                    "Plot and Events", 
                    "Themes and Motifs",
                    "Setting and Context",
                    "Writing Style and Techniques"
                ],
                "question_types": [
                    "Character Analysis",
                    "Plot Summary", 
                    "Theme Exploration",
                    "Context Understanding",
                    "Style Analysis"
                ]
            },
            "comprehension": {
                "categories": [
                    "Basic Facts",
                    "Character Details",
                    "Event Sequences",
                    "Key Quotes",
                    "Relationships"
                ],
                "question_types": [
                    "Who/What/When/Where",
                    "Character Description",
                    "Event Ordering",
                    "Quote Identification",
                    "Relationship Mapping"
                ]
            }
        }
    
    def get_template(self, template_name):
        """Get a specific taxonomy template."""
        return self.templates.get(template_name, self.templates["literary_analysis"])
    
    def create_taxonomy_prompt(self, template_name, book_title):
        """Create a structured prompt for taxonomy generation."""
        template = self.get_template(template_name)
        
        prompt = f"""
        Create a structured knowledge taxonomy for '{book_title}' using this template:

        CATEGORIES:
        {chr(10).join(f"- {cat}" for cat in template['categories'])}

        For each category, provide:
        1. Key elements/concepts
        2. Important details
        3. Relationships between elements

        Format the response as a structured outline with clear categories and subpoints.
        """
        
        return prompt
    
    def create_question_generation_prompt(self, taxonomy, template_name, num_questions=10):
        """Create a structured prompt for question generation."""
        template = self.get_template(template_name)
        
        prompt = f"""
        Based on this taxonomy for Pride and Prejudice:

        {taxonomy}

        Generate {num_questions} diverse questions covering these question types:
        {chr(10).join(f"- {qtype}" for qtype in template['question_types'])}

        Requirements:
        1. Each question should be specific and answerable from the text
        2. Distribute questions across all categories
        3. Include different difficulty levels (easy, medium, hard)
        4. Format as numbered list
        5. Make questions varied in format (who, what, how, why, compare, analyze)
        """
        
        return prompt


class DatasetCreator:
    """Main class for creating instruction fine-tuning datasets."""

    def __init__(self):
        self.gemini = GeminiProClient()
        self.rag = PridePrejudiceRAG()
        self.taxonomy_template = TaxonomyTemplate()

    def generate_knowledge_taxonomy(self, template_name="literary_analysis"):
        """Generate a structured knowledge taxonomy using templates."""
        prompt = self.taxonomy_template.create_taxonomy_prompt(template_name, "Pride and Prejudice")
        
        print(f"📚 Generating {template_name} taxonomy...")
        taxonomy = self.gemini.generate_response(prompt)

        if taxonomy:
            print("✅ Taxonomy generated successfully")
            return taxonomy
        else:
            print("❌ Failed to generate taxonomy")
            return None

    def generate_questions(self, taxonomy, template_name="literary_analysis", num_questions=10):
        """Generate structured questions using taxonomy templates."""
        prompt = self.taxonomy_template.create_question_generation_prompt(
            taxonomy, template_name, num_questions
        )

        print(f"❓ Generating {num_questions} questions using {template_name} template...")
        questions = self.gemini.generate_response(prompt)

        if questions:
            print("✅ Questions generated successfully")
            return questions
        else:
            print("❌ Failed to generate questions")
            return None
    
    def generate_multiple_question_sets(self, taxonomy, num_sets=3):
        """Generate multiple sets of questions using different templates."""
        all_questions = []
        
        templates = ["literary_analysis", "comprehension"]
        
        for i, template in enumerate(templates):
            questions = self.generate_questions(taxonomy, template, 5)
            if questions:
                all_questions.append(f"\n--- {template.upper()} QUESTIONS ---\n{questions}")
        
        return "\n".join(all_questions) if all_questions else None


class SystemTester:
    """Simple testing suite for system components."""

    def __init__(self):
        self.gemini = GeminiProClient()
        self.rag = PridePrejudiceRAG()

    def test_gemini_pro(self):
        """Test Gemini Pro API."""
        response = self.gemini.generate_response("What is 2 + 2?")
        return response is not None

    def test_rag_system(self):
        """Test RAG system."""
        if not self.rag.qa_chain:
            return False
            
        test_questions = [
            "Who is Elizabeth Bennet?",
            "What happens at the first ball?",
            "What are the main themes?",
            "Who is Mr. Darcy?",
            "What is the setting of the story?"
        ]
        
        for question in test_questions:
            response = self.rag.query(question)
            if not response or response == "RAG system not initialized":
                return False
                
        return True

    def run_tests(self):
        """Run all tests."""
        print("Testing system components...")
        
        gemini_ok = self.test_gemini_pro()
        rag_ok = self.test_rag_system()
        
        print(f"Gemini Pro: {'✅' if gemini_ok else '❌'}")
        print(f"RAG System: {'✅' if rag_ok else '❌'}")
        
        return gemini_ok and rag_ok



"""Main execution function."""

def main():
    print("=" * 60)
    print("PRIDE AND PREJUDICE DATASET CREATION")
    print("=" * 60)

    # Run tests first
    tester = SystemTester()
    if not tester.run_tests():
        print("\n❌ System tests failed. Exiting.")
        return

    print("\n✅ All tests passed! Starting dataset creation...")

    # Create dataset
    creator = DatasetCreator()

    # Generate taxonomy using literary analysis template
    taxonomy = creator.generate_knowledge_taxonomy("literary_analysis")
    if not taxonomy:
        return

    # Generate multiple question sets using different templates
    print("\n📝 Generating multiple question sets...")
    all_questions = creator.generate_multiple_question_sets(taxonomy)
    if not all_questions:
        return

    # Save results to file
    with open("pride_prejudice_taxonomy.md", "w") as f:
        f.write("# Pride and Prejudice Knowledge Taxonomy\n\n")
        f.write(taxonomy)
        f.write("\n\n# Generated Questions\n\n")
        f.write(all_questions)
    
    print("\n💾 Results saved to pride_prejudice_taxonomy.md")
    print("\n🎉 Dataset creation completed!")

main()
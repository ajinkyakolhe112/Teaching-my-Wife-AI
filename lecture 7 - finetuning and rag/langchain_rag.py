"""
Simple RAG implementation using LangChain for question answering.
"""

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import load_dataset

class SimpleRAG:
    def __init__(self, num_books=100):
        """Initialize RAG system with specified number of books."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self._setup_vector_store(num_books)
    
    def _setup_vector_store(self, num_books):
        """Create vector store from Project Gutenberg books."""
        dataset = load_dataset("manu/project_gutenberg", split="train")
        texts = [book['text'] for book in dataset[:num_books]]
        chunks = self.text_splitter.create_documents(texts)
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def answer_question(self, question):
        """Retrieve relevant passages for a given question."""
        return self.retriever.get_relevant_documents(question)

def main():
    rag = SimpleRAG()
    questions = [
        "What is the meaning of life?",
        "Tell me about love and romance",
        "What is the role of nature in human existence?",
        "How do people deal with loss and grief?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        docs = rag.answer_question(question)
        for i, doc in enumerate(docs, 1):
            print(f"\nPassage {i}: {doc.page_content[:200]}...")

if __name__ == "__main__":
    main() 
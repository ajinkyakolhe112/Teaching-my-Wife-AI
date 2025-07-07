import requests
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# Step 1: Document Loading - Get text from external source
text = requests.get("https://www.gutenberg.org/cache/epub/1342/pg1342.txt").text
documents = [Document(page_content=text)]

# Step 2: Text Chunking - Split large documents into smaller, manageable pieces
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Step 3: Embedding Generation - Convert text chunks into numerical vectors
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Step 4: Vector Storage - Store embeddings in a searchable vector database
vectorstore = FAISS.from_documents(chunks, embeddings)

# Step 5: LLM Setup - Initialize the language model for text generation
llm = HuggingFacePipeline(pipeline=pipeline("text-generation", model="microsoft/DialoGPT-medium", max_length=512))

# Step 6: RAG Chain Creation - Combine retrieval and generation into a single chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(search_kwargs={"k": 3}))

# Step 7: Query Processing - Ask questions and get answers using the RAG system
question = "Who is the main protagonist in Pride and Prejudice and what is her family situation?"
response = qa_chain.run(question)
print(f"Question: {question}")
print(f"Answer: {response}")
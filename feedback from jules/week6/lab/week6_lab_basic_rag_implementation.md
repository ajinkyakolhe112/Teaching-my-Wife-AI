# Lab 6: Building a Basic RAG System with LangChain

**Objective:** To provide hands-on experience building a simple Retrieval Augmented Generation (RAG) pipeline using LangChain, an open-source embedding model, a local vector store (FAISS), and a local LLM via Ollama.

**Prerequisites:**
*   Python 3.x installed.
*   Ollama installed with a model like `phi3:mini` (recommended for this lab due to speed) or `llama3:8b` pulled.
    *   To pull `phi3:mini`: `ollama pull phi3:mini`
*   Necessary Python libraries. You can install them using pip:
    ```bash
    pip install langchain langchain_community sentence-transformers faiss-cpu
    # faiss-cpu is for a CPU-only FAISS build. If you have a GPU and want to use it for FAISS, you might install faiss-gpu.
    # For this lab, faiss-cpu is sufficient.
    ```

---

## Tasks:

### 1. Setup and Installations

*   Ensure you have Python and Ollama installed.
*   Make sure your chosen Ollama model (e.g., `phi3:mini`) is running or available. You can typically start Ollama by running the Ollama application.
*   Install the required Python libraries by running the pip command from the prerequisites section in your terminal.

### 2. Prepare Sample Documents

We'll create a few simple text files with distinct information.

*   **Create the following files in your working directory:**

    *   **`doc1.txt`:**
        ```text
        Paris is the capital and most populous city of France. 
        It is situated on the Seine River, in the north of the country. 
        The city is known for its art, culture, and landmarks, including the Eiffel Tower and the Louvre Museum.
        Paris is often called the "City of Light".
        ```

    *   **`doc2.txt`:**
        ```text
        Berlin is the capital and largest city of Germany by both area and population.
        Its 3.7 million inhabitants make it the European Union's most populous city.
        Berlin is known for its historical associations, internationalism and tolerance, lively nightlife, its many cafes, clubs, bars, street art, and numerous museums, palaces, and other sites of historic interest.
        The Brandenburg Gate is a famous landmark in Berlin.
        ```

    *   **`doc3.txt`:**
        ```text
        Rome is the capital city of Italy. It is also the country's most populated comune.
        Rome is located in the central-western portion of the Italian Peninsula, within Lazio (Latium), along the shores of the Tiber.
        The city is famous for its ancient history, with landmarks such as the Colosseum, Roman Forum, and Pantheon.
        Vatican City, an independent country, is an enclave within Rome.
        ```

### 3. Choose a RAG Framework

For this lab, we will use **LangChain**. LangChain provides a comprehensive framework for building applications with LLMs, including RAG pipelines.

### 4. Building the RAG Pipeline (using LangChain)

Let's build the RAG pipeline step by step.

```python
# Import necessary LangChain components
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

import os

# --- Create dummy files if they don't exist (for notebook convenience) ---
# Normally, you would have these files ready.
doc_contents = {
    "doc1.txt": """Paris is the capital and most populous city of France. 
It is situated on the Seine River, in the north of the country. 
The city is known for its art, culture, and landmarks, including the Eiffel Tower and the Louvre Museum.
Paris is often called the "City of Light".""",
    "doc2.txt": """Berlin is the capital and largest city of Germany by both area and population.
Its 3.7 million inhabitants make it the European Union's most populous city.
Berlin is known for its historical associations, internationalism and tolerance, lively nightlife, its many cafes, clubs, bars, street art, and numerous museums, palaces, and other sites of historic interest.
The Brandenburg Gate is a famous landmark in Berlin.""",
    "doc3.txt": """Rome is the capital city of Italy. It is also the country's most populated comune.
Rome is located in the central-western portion of the Italian Peninsula, within Lazio (Latium), along the shores of the Tiber.
The city is famous for its ancient history, with landmarks such as the Colosseum, Roman Forum, and Pantheon.
Vatican City, an independent country, is an enclave within Rome."""
}

# Create a directory for documents if it doesn't exist
doc_dir = "./sample_docs/"
if not os.path.exists(doc_dir):
    os.makedirs(doc_dir)

for filename, content in doc_contents.items():
    with open(os.path.join(doc_dir, filename), "w") as f:
        f.write(content)

print(f"Created sample documents in '{doc_dir}' directory.")


# --- Step 1: Load Documents ---
# We'll use DirectoryLoader to load all .txt files from our sample_docs directory
print("\\n--- Loading Documents ---")
loader = DirectoryLoader(doc_dir, glob="*.txt", loader_cls=TextLoader)
documents = loader.load()
print(f"Loaded {len(documents)} documents.")
for i, doc in enumerate(documents):
    print(f"Document {i+1} source: {doc.metadata.get('source', 'N/A')}, Preview: {doc.page_content[:100]}...")


# --- Step 2: Split Documents (Chunking) ---
print("\\n--- Splitting Documents into Chunks ---")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,  # Maximum size of a chunk (in characters)
    chunk_overlap=50   # Number of characters to overlap between chunks
)
# This splitter tries to keep paragraphs, then sentences, then words together.
chunks = text_splitter.split_documents(documents)
print(f"Split documents into {len(chunks)} chunks.")
if chunks:
    print(f"Example chunk 0: {chunks[0].page_content}")
    print(f"Example chunk 0 metadata: {chunks[0].metadata}")


# --- Step 3: Initialize Embedding Model ---
print("\\n--- Initializing Embedding Model ---")
# We'll use a popular open-source sentence transformer model.
# The first time you run this, it might download the model (a few hundred MB).
embedding_model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
print(f"Loaded embedding model: {embedding_model_name}")


# --- Step 4: Create Vector Store ---
print("\\n--- Creating Vector Store ---")
# Embed the document chunks and store them in a FAISS vector store.
# FAISS is a library for efficient similarity search on dense vectors.
try:
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("Vector store created successfully using FAISS.")
except Exception as e:
    print(f"Error creating FAISS vector store: {e}")
    vector_store = None


# --- Step 5: Initialize LLM ---
print("\\n--- Initializing LLM ---")
# Connect to a local model running via Ollama (e.g., phi3:mini)
# Make sure Ollama is running and the model is pulled (e.g., `ollama pull phi3:mini`)
OLLAMA_MODEL = "phi3:mini" # Change if you are using a different model like llama3:8b
try:
    llm = Ollama(model=OLLAMA_MODEL)
    print(f"Connected to Ollama model: {OLLAMA_MODEL}")
    # Test with a simple prompt
    # print(f"Testing Ollama LLM: {llm.invoke('Say hi!')}")
except Exception as e:
    print(f"Error initializing Ollama LLM. Is Ollama running with model '{OLLAMA_MODEL}'? Error: {e}")
    llm = None


# --- Step 6: Create Retriever ---
print("\\n--- Creating Retriever ---")
if vector_store:
    retriever = vector_store.as_retriever(
        search_type="similarity", # Other options: "mmr", "similarity_score_threshold"
        search_kwargs={"k": 2}    # Retrieve top 2 most relevant chunks
    )
    print("Retriever created from vector store.")
    # Example: Test retriever
    # sample_query_for_retriever = "What is Paris known for?"
    # retrieved_docs = retriever.invoke(sample_query_for_retriever)
    # print(f"Retrieved docs for '{sample_query_for_retriever}':")
    # for doc in retrieved_docs:
    #     print(f"- {doc.page_content[:100]}...")
else:
    print("Skipping retriever creation as vector store is not available.")
    retriever = None


# --- Step 7: Create RAG Chain (More Explicit Method) ---
print("\\n--- Creating RAG Chain ---")
if llm and retriever:
    # Define a prompt template. This guides the LLM on how to use the context.
    template = """Answer the question based only on the following context:

    {context}

    Question: {question}

    Answer:"""
    prompt = PromptTemplate.from_template(template)

    # This chain will:
    # 1. Take the user's question.
    # 2. Use the retriever to fetch relevant context.
    # 3. Format the prompt with the question and context.
    # 4. Pass the formatted prompt to the LLM.
    # 5. Parse the LLM's output.
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} # Pass question to retriever and along
        | prompt
        | llm
        | StrOutputParser() # Parses the LLM output into a string
    )
    print("RAG chain created successfully.")
else:
    print("Skipping RAG chain creation due to missing LLM or retriever.")
    rag_chain = None

# --- Alternative: Using RetrievalQA chain (simpler but less flexible) ---
# if llm and retriever:
#    qa_chain = RetrievalQA.from_chain_type(
#        llm=llm,
#        chain_type="stuff", # "stuff" puts all retrieved docs into context. Other types: "map_reduce", "refine"
#        retriever=retriever,
#        return_source_documents=True # Optionally return the source documents
#    )
#    print("RetrievalQA chain created.")
# else:
#    print("Skipping RetrievalQA chain creation.")
#    qa_chain = None

```

**Explanation of Parameters:**
*   `RecursiveCharacterTextSplitter`:
    *   `chunk_size`: Defines the maximum size of each chunk. If a paragraph or sentence is larger than this, it will be split.
    *   `chunk_overlap`: Maintains some overlap between consecutive chunks. This helps preserve context that might otherwise be lost at the boundary of a split.
*   `HuggingFaceEmbeddings`:
    *   `model_name`: Specifies which sentence transformer model to use from Hugging Face Hub. `all-MiniLM-L6-v2` is a good general-purpose, lightweight model.
*   `FAISS.from_documents`:
    *   Takes the document chunks and the embedding model to create a FAISS index in memory.
*   `Ollama`:
    *   `model`: The name of the Ollama model you want to use (e.g., `phi3:mini`).
*   `vector_store.as_retriever()`:
    *   `search_kwargs={"k": 2}`: Tells the retriever to fetch the top 2 most similar document chunks for a given query.
*   `PromptTemplate`:
    *   The template string uses `{context}` and `{question}` as placeholders. LangChain will fill these in. The instruction "Answer the question based only on the following context" is crucial for guiding the RAG system.

### 5. Querying the RAG System

Now, let's ask some questions.

```python
if rag_chain:
    print("\\n--- Querying the RAG System ---")
    
    query1 = "What is the capital of France?"
    print(f"Query 1: {query1}")
    response1 = rag_chain.invoke(query1)
    print(f"Response 1: {response1}")

    query2 = "What is Berlin known for?"
    print(f"\\nQuery 2: {query2}")
    response2 = rag_chain.invoke(query2)
    print(f"Response 2: {response2}")

    query3 = "Where is Rome located?"
    print(f"\\nQuery 3: {query3}")
    response3 = rag_chain.invoke(query3)
    print(f"Response 3: {response3}")

    # Optional: Inspect retrieved documents for a query
    # (Requires a bit more setup if using the Runnable chain directly,
    # or using `return_source_documents=True` with RetrievalQA)
    if retriever:
        print("\\n--- Inspecting Retrieved Documents for Query 1 ---")
        retrieved_docs_for_query1 = retriever.invoke(query1)
        for i, doc in enumerate(retrieved_docs_for_query1):
            print(f"Retrieved Doc {i+1} for Query 1 (Source: {doc.metadata.get('source')}):\\n{doc.page_content}\\n---")
            
else:
    print("\\nSkipping querying as RAG chain was not created.")

# If using RetrievalQA chain and want to see source documents:
# if qa_chain:
#    result = qa_chain.invoke({"query": query1})
#    print(f"Answer: {result['result']}")
#    print("Source Documents:")
#    for doc in result['source_documents']:
#        print(f"- {doc.page_content[:100]}... (Source: {doc.metadata.get('source')})")
```

### 6. Experiment: Querying for Information Not in Documents

Let's see how the RAG system responds when asked a question whose answer is not in our small document set.

```python
if rag_chain:
    print("\\n--- Experiment: Querying for Unavailable Information ---")
    
    query_unavailable = "What is the currency of Japan?"
    print(f"Query (Unavailable Info): {query_unavailable}")
    response_unavailable = rag_chain.invoke(query_unavailable)
    print(f"Response (Unavailable Info): {response_unavailable}")
else:
    print("\\nSkipping experiment as RAG chain was not created.")
```

**Expected Observation:**
Ideally, the LLM, guided by the prompt "Answer the question based *only* on the following context," should indicate that the information is not available in the provided context, or it might give a very generic answer if the retriever found no relevant chunks. It should avoid making up an answer about Japan's currency if the context provided was about European cities.

---

**Lab Conclusion:**
In this lab, you built a basic RAG pipeline using LangChain. You learned how to:
*   Load and chunk documents.
*   Use an open-source embedding model.
*   Store and retrieve document embeddings from a FAISS vector store.
*   Connect to a local LLM via Ollama.
*   Construct a RAG chain to answer questions based on retrieved context.

This forms the foundation for building more sophisticated RAG applications. You can experiment further by adding more documents, trying different embedding models, adjusting chunking strategies, or using different LLMs.

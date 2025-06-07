# Assignment 6: Extending Your RAG System

**Objective:** To build upon the lab by adding more documents, experimenting with different components, or analyzing the RAG system's behavior.

**Prerequisites:**
*   Completion of Lab 6: Building a Basic RAG System with LangChain.
*   Python 3.x installed.
*   Ollama installed with a model like `phi3:mini` or `llama3:8b` pulled.
*   Libraries from Lab 6: `langchain`, `langchain_community`, `sentence-transformers`, `faiss-cpu`.

---

## Instructions:

**Choose ONE or TWO of the following tasks to implement.**
Your submission should clearly state which task(s) you chose, detail your methodology, show your code changes (if any), present the results (LLM outputs, observations), and include a discussion of your findings.

---

### Task Options:

#### 1. Add More Documents

*   **Objective:** To expand the knowledge base of your RAG system and test its ability to retrieve information from a larger set of documents.
*   **Methodology:**
    1.  **Find or Create New Documents:**
        *   Gather 5-10 additional text documents (.txt files) on a specific topic of interest.
        *   Examples: Short biographies of different scientists, basic facts about different programming languages, summaries of different chapters of a book, news articles on a particular theme.
        *   Ensure the content is distinct enough to test retrieval.
        *   Place these new documents in your `sample_docs` directory (or a new directory if you prefer, and update your `DirectoryLoader` path).
    2.  **Integrate Documents:**
        *   Modify your lab script to load all documents (the original 3 + your new ones).
        *   Re-build the vector store with all document chunks.
    3.  **Test the System:**
        *   Craft at least 3 new questions that should be answerable *only* from your newly added documents.
        *   Craft 1-2 questions that should still be answerable from the original documents.
        *   (Optional) Craft 1 question that spans information from both old and new document sets if possible.
*   **Report:**
    *   **New Data:** Briefly describe the topic of your new documents and list the filenames. Provide the content of 1-2 example new documents.
    *   **Queries and Results:** For each test query:
        *   State the query.
        *   Show the LLM's response from your RAG system.
        *   (Optional but recommended) Show the source documents retrieved by the retriever for one of your new queries.
    *   **Discussion:**
        *   Did the RAG system successfully answer questions based on the new documents?
        *   Did its performance on questions related to the original documents change?
        *   Any interesting observations or challenges encountered?

---

#### 2. Experiment with a Different Embedding Model

*   **Objective:** To observe how changing the embedding model affects retrieval and the final answer quality.
*   **Methodology:**
    1.  **Research an Embedding Model:**
        *   Go to the [Hugging Face Model Hub](https://huggingface.co/models) and filter for "Sentence Transformers" or search for models like `BAAI/bge-small-en-v1.5` (a popular alternative to `all-MiniLM-L6-v2`) or `sentence-transformers/paraphrase-MiniLM-L3-v2`.
        *   Choose one that is different from `all-MiniLM-L6-v2` used in the lab. Note its Hugging Face identifier.
    2.  **Modify RAG Pipeline:**
        *   In your lab script, change the `embedding_model_name` in `HuggingFaceEmbeddings` to your newly chosen model.
        *   Re-build the vector store using this new embedding model (this will re-embed all your existing documents).
    3.  **Compare Performance:**
        *   Use the **same set of 2-3 questions** from the original lab (e.g., "What is the capital of France?", "What is Berlin known for?").
        *   Run these questions through the RAG system with the original `all-MiniLM-L6-v2` embeddings. Record the answers and (if possible) the retrieved chunks.
        *   Run the same questions through the RAG system with your **new** embedding model. Record the answers and (if possible) the retrieved chunks.
*   **Report:**
    *   **New Embedding Model:** State the Hugging Face identifier of the model you chose.
    *   **Comparison:** For each test query:
        *   Query: `[Your Query]`
        *   Response (Original Embeddings - `all-MiniLM-L6-v2`): `[LLM Output]`
        *   Retrieved Chunks (Original, if you can inspect them): `[Brief summary or content of top 1-2 chunks]`
        *   Response (New Embeddings - `your-chosen-model`): `[LLM Output]`
        *   Retrieved Chunks (New, if you can inspect them): `[Brief summary or content of top 1-2 chunks]`
    *   **Discussion:**
        *   Did you notice any difference in the quality or relevance of the retrieved chunks with the new embedding model?
        *   Did the final LLM answers change significantly? Were they better, worse, or just different?
        *   Was one embedding model noticeably slower to load or use for embedding? (This might be hard to measure without proper benchmarking but note any subjective experience).

---

#### 3. Experiment with `k` (Number of Retrieved Documents)

*   **Objective:** To understand the impact of retrieving more or fewer document chunks on the LLM's ability to answer questions.
*   **Methodology:**
    1.  **Choose a Question:** Select one reasonably complex question from your lab or from the "Add More Documents" task (if you did that) whose answer might benefit from seeing multiple pieces of context, or where too much context could be confusing.
        *   Example: "Compare and contrast Paris and Berlin based on the provided documents." (You'd need to ensure your documents have comparable info for this).
        *   Or, a simpler question: "Tell me everything you know about Paris based on the documents."
    2.  **Modify Retriever Settings:**
        *   In your lab script, find the line where the retriever is created: `retriever = vector_store.as_retriever(search_kwargs={"k": K_VALUE})`.
        *   Run your chosen question through the RAG system three times, setting `K_VALUE` to:
            *   `k=1` (retrieve only the single most relevant chunk)
            *   `k=2` (as in the lab)
            *   `k=3` or `k=5` (retrieve more chunks)
    3.  **Record Results:** For each value of `k`:
        *   State the value of `k`.
        *   Show the LLM's response to your chosen question.
        *   (Highly recommended) List or summarize the content of the chunks retrieved for that `k` value.
*   **Report:**
    *   **Chosen Question:** `[Your chosen question]`
    *   **Results for k=1:**
        *   Retrieved Chunks: `[Summary/content of chunk(s)]`
        *   LLM Response: `[LLM Output]`
    *   **Results for k=2 (or your lab default):**
        *   Retrieved Chunks: `[Summary/content of chunk(s)]`
        *   LLM Response: `[LLM Output]`
    *   **Results for k=3 (or k=5):**
        *   Retrieved Chunks: `[Summary/content of chunk(s)]`
        *   LLM Response: `[LLM Output]`
    *   **Discussion:**
        *   How did the LLM's answer change as `k` increased?
        *   Was there a point where retrieving more documents seemed to improve the answer?
        *   Was there a point where retrieving more documents seemed to confuse the LLM or make the answer worse (e.g., by including less relevant information)?
        *   What are the potential trade-offs of a very small `k` versus a very large `k` (consider context window limits, noise, completeness)?

---

#### 4. Analyze a "Failure Case"

*   **Objective:** To critically examine the limitations of your RAG system and think about potential improvements.
*   **Methodology:**
    1.  **Find a Failure Case:**
        *   Use your RAG system from the lab (with the original 3 documents or with more if you did Task 1).
        *   Try different questions until you find one where the system performs poorly. "Poorly" could mean:
            *   The answer is factually incorrect, but the correct information *is* present in your documents.
            *   The system says it cannot answer, but the information *is* present.
            *   The system's answer is nonsensical or hallucinates information not supported by the retrieved context.
            *   The retrieved context is clearly wrong or irrelevant to the question.
    2.  **Describe the Failure:**
        *   State the exact question you asked.
        *   Show the (incorrect or poor) response from the LLM.
        *   Show the actual context/chunks that were retrieved by the retriever for this question.
        *   Explain why the LLM's response is a failure (e.g., "The correct answer is X, found in doc Y, but the LLM said Z," or "The retrieved chunks were about A and B, but the question was about C").
    3.  **Hypothesize the Cause:**
        *   Why do you think the system failed for this specific case? Consider:
            *   **Retrieval Issue:** Was the wrong context retrieved? Were the right documents not ranked highly enough? Is the query ambiguous leading to poor embedding similarity?
            *   **Generation Issue:** Was the correct context retrieved, but the LLM ignored it, misinterpreted it, or over-relied on its pre-trained knowledge?
            *   **Chunking Issue:** Is the information fragmented across chunks in a way that makes it hard to piece together?
            *   **Prompting Issue:** Is the prompt template for the LLM not guiding it well enough to use the context effectively or to admit when it doesn't know?
    4.  **Suggest a Potential Improvement:**
        *   Based on your hypothesis, suggest **one specific change** you could make to the RAG system (e.g., to the chunking strategy, embedding model, retriever settings, LLM prompt template) that *might* help address this failure. You do not need to implement the fix, just describe it.
*   **Report:**
    *   **Question Asked:** `[Your query]`
    *   **LLM's (Failure) Response:** `[LLM Output]`
    *   **Retrieved Context (Summarize or show key parts):** `[Retrieved Chunks]`
    *   **Explanation of Failure:** `[Your explanation]`
    *   **Hypothesized Cause(s):** `[Your hypothesis]`
    *   **Suggested Potential Improvement:** `[Your suggestion]`

---

## Submission Guidelines:

*   Clearly indicate which **Task(s) (1, 2, 3, or 4)** you chose. If you did more than one, present them sequentially.
*   Compile your responses, methodology, code snippets (if you made changes to the lab code), results (LLM outputs, data descriptions), and discussion into a **single document**.
*   You can use Markdown (preferred, save as a `.md` file) or create a PDF.
*   Name your file clearly, e.g., `week6_assignment_yourname.md`.

This assignment encourages you to explore the RAG system more deeply and think critically about its components and behavior. Good luck!

# Project Summary  
## Smart Contract Summary & Q&A Assistant

### 1. Project Overview

This project is a simplified end-to-end Retrieval-Augmented Generation (RAG) application that allows users to:

- Upload long documents (PDF / DOCX / TXT)
- Automatically extract and chunk text
- Generate embeddings
- Store them in a local FAISS vector database
- Ask natural language questions about the document
- Receive grounded answers with source chunk references

The system demonstrates practical implementation of LLM pipelines using LangChain.

---

### 2. Why I Used Z.ai (GLM-4.7)

I selected **Z.ai (GLM-4.7)** for the following reasons:

1. I already have an active subscription.
2. It provides OpenAI-compatible API support.
3. It offers strong reasoning capability suitable for document Q&A.
4. It integrates smoothly with LangChain’s `ChatOpenAI` wrapper.
5. It reduces additional cost overhead since I already invested in it.

Using Z.ai allowed me to:
- Avoid switching providers mid-project
- Maintain consistency in experimentation
- Focus on pipeline engineering instead of provider configuration

---

### 3. Why I Used FAISS

FAISS was selected because:

1. It runs locally (no external database required).
2. It is lightweight and fast for small-to-medium document sizes.
3. It is widely used in research and production RAG systems.
4. It integrates natively with LangChain.
5. It does not require server deployment for this workshop scope.

For this project’s scale, FAISS provides:
- Fast similarity search
- Minimal configuration
- Good reproducibility for evaluation

---

### 4. Why Local Embeddings (SentenceTransformers)

Instead of using API-based embeddings, I used:

`sentence-transformers/all-MiniLM-L6-v2`

Reasons:
- Avoid API model mismatch issues.
- No dependency on provider-specific embedding endpoints.
- Faster ingestion.
- More stable during demonstration.
- Fully offline embedding computation.

This choice improves reliability during evaluation.

---

### 5. System Architecture

Upload → Text Extraction → Chunking → Embeddings → FAISS  
→ Retriever → LLM (Z.ai GLM-4.7) → Answer + Citations

---

### 6. Key Concepts Demonstrated

- RAG architecture
- Text chunking strategies
- Vector similarity search
- Prompt engineering with context injection
- Streaming LLM responses
- Gradio UI integration
- Local persistence of vectorstore
- Clean minimal architecture design

---

### 7. Design Philosophy

This project intentionally uses a **single-file architecture** to:

- Keep the system easy to explain during examination
- Avoid unnecessary abstraction layers
- Demonstrate understanding of the complete pipeline
- Maintain clarity in execution flow

---

### 8. Limitations

- English-only documents
- Not optimized for very large datasets
- No production-level security controls
- Not a legal advisory system

---

### 9. Disclaimer

This application is a technical demonstration and does not provide legal advice.
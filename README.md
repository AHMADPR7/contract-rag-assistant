# Smart Contract Summary & Q&A Assistant

A minimal end-to-end RAG application built using:

- LangChain
- Z.ai (GLM-4.7)
- FAISS
- SentenceTransformers
- Gradio

This project allows users to upload contracts and interact with them using conversational Q&A.

---

## Features

- Upload PDF / DOCX / TXT
- Automatic text extraction
- Smart chunking
- Local embedding generation
- FAISS vector search
- Context-grounded LLM responses
- Streaming chat interface
- Source chunk references
- Fully local vectorstore

---

## Technology Stack

| Component | Tool Used | Reason |
|------------|------------|------------|
| LLM | Z.ai GLM-4.7 | Already subscribed, OpenAI-compatible, strong reasoning |
| Embeddings | SentenceTransformers | Local, stable, no API dependency |
| Vector Store | FAISS | Fast, lightweight, local |
| UI | Gradio | Quick interactive interface |
| Framework | LangChain | Modular LLM pipeline construction |

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
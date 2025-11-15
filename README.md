# Gemma 3 QA Fine-Tuning (SQuAD)
Experiments comparing **Full FT**, **LoRA (PEFT)**, and **Layer Freezing** for **extractive QA** on **SQuAD 1.1** with **Gemma 3 (~4B)**. Designed for a single GPU (RTX 3070 Ti, 8 GB).


## Dependencies installation:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync
```

# RAG-Powered Agent

This project implements a Retrieval-Augmented Generation (RAG) agent using modern LLM tooling and vector search technologies. It combines OpenAI models with LangChain workflows and FAISS-based semantic search for fast, high-quality retrieval.

## Technology Stack

* **OpenAI API** – LLMs for generation and reasoning
* **LangChain / LangGraph** – Agentic RAG orchestration
* **Hugging Face Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **FAISS** – High-performance vector store for similarity search

---

## ▶️ Getting Started

Follow the steps below to set up and run the RAG agent locally.

### 1. 🔑 Obtain an OpenAI API Key

The agent uses the OpenAI provider.
Get your API key from:

➡️ [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)

Then copy the example environment file and add your key:

```bash
cp env.example .env
```

Open `.env` and set:

```
OPENAI_API_KEY=your_api_key_here
```

---

### 2. 📦 Install Dependencies

This project uses **uv** for fast Python package management.
Install the required packages:

```bash
uv install
```

---

### 3. ▶️ Run the Project

Start the agent:

```bash
uv run src/main.py
```
# Graph-RAG-Framework

## 📚 GraphRAG with Neo4j + LangChain

This project builds a **Graph-RAG (Retrieval Augmented Generation)** knowledge base from your documents.  
It extracts a knowledge graph using an LLM, stores it in **Neo4j**, generates embeddings, and creates **hybrid search indexes** (vector + keyword) for powerful question-answering.

---

## ✨ Features

1. **Document Pre-processing**  
   - Load PDFs (and optionally `.txt` / `.md`).
   - Clean and chunk text for downstream processing.

2. **Knowledge Graph Creation**  
   - Convert chunks to entities and relationships using a Large Language Model (LLM).
   - Store nodes and relationships directly in Neo4j.

3. **Vector & Keyword Indexing**  
   - Compute embeddings and store them as node properties.
   - Build vector and full-text indexes in Neo4j for similarity or hybrid search.

4. **Graph-Aware QA** *(Optional)*  
   - Natural-language questions are translated into Cypher queries with an LLM.
   - Execute Cypher against the graph and return text answers.

---

## 🗂️ Project Structure
.
├─ components/
│  ├─ embeddings.py      -- GetEmbeddings: returns a LangChain embeddings model (OpenAI, Cohere, etc.)
│  ├─ llms.py            -- GetLLM: returns a LangChain chat model (OpenAI, Anthropic, Google Gemini…)
│  ├─ neo4j_store.py     -- Neo4jStore: connect/clear/add docs/create hybrid indexes + similarity/hybrid search
│  └─ graph_ops.py       -- create_knowledge_graph() and cypher_qa() utilities (require an initialized LLM)
│
├─ options/
│  ├─ base_options.py    -- Base CLI arguments: API keys, Neo4j URI, DB name, etc.
│  └─ train_options.py   -- Training/ingestion options: chunk size, LLM provider/model, embedding model…
│
├─ utils/
│  └─ preprocessing.py   -- DocumentProcessor: loads & cleans files, splits into chunks with metadata
│
├─ train.py              -- Main entry point: runs the full pipeline
├─ requirements.txt      -- Python dependencies for the project
└─ README.md


### 🛠️ Framework Ready
This project is designed as a **framework-agnostic starter kit**.  

You can easily extend it to:

- **FastAPI Service**: Expose endpoints (e.g., ingest documents, query Neo4j) to turn it into a web API.
- **n8n Automation**: Connect to Neo4j and run Graph RAG workflows as part of no-code automation.

This flexibility allows you to scale from a simple command-line script to a full **API as a Service** or automated pipeline.


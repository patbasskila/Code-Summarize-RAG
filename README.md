Code-Summarize-RAG 

A Retrieval-Augmented Generation (RAG) system that automatically analyzes GitHub repositories, retrieves semantically relevant code, and generates high-quality AI summaries using transformer embeddings, FAISS vector search, and OpenAI models. 

This project demonstrates full-stack AI engineering — combining NLP, vector search, backend systems, and applied ML — suitable for both hiring teams and graduate admissions committees. 

 

✨ Overview 

Code-Summarize-RAG is an end-to-end platform that: 

- Fetches and analyzes public GitHub repositories 

- Extracts source code across multiple languages 

- Embeds code files using MiniLM-L6-v2 (Sentence Transformers) 

- Builds a FAISS index for high-speed semantic retrieval 

- Uses GPT-4o to generate: 

  - File-level summaries 

  - A combined high-level project summary 

- Stores processed results in /tmp/processed_repos.json 

- Displays summaries in an interactive web dashboard 

 

🚀 Key Features 

Retrieval-Augmented Generation 

- Transformer-based embeddings 

- Semantic vector search (FAISS) 

- Retrieval of most relevant files 

- Context-optimized LLM summarization 

AI Summaries 

- Concise 1–2 sentence file summaries 

- 2–3 sentence repository-level summary 

- Automatic detection of frameworks & technologies 

Full-Stack System 

- Flask backend with async job orchestration 

- Threaded workers for non-blocking processing 

- HTML/JS UI with searchable summaries 

- Clean module separation for extensibility 

Academic + Professional Relevance 

Demonstrates skills in: 

- Information retrieval 

- Modern NLP and embeddings 

- Vector databases 

- Web backend engineering 

- Applied AI/ML system design


🧩 Architecture
High-Level System Diagram

flowchart LR
    A[User Submits GitHub URL] --> B[Flask Backend]
    B --> C[Async Worker Thread]
    C --> D[GitHub API<br/>Fetch Code]
    D --> E[Embedding Model<br/>MiniLM-L6-v2]
    E --> F[FAISS Vector Index]
    F --> G[Retrieve Top-k Code Files]
    G --> H[GPT-4o Summaries]
    H --> I[Final Combined Summary]
    I --> J[Store JSON]
    J --> K[Dashboard UI]


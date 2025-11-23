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


## Architecture High-Level System Diagram

```mermaid
flowchart LR
  A[User Submits GitHub URL] --> B[Flask Backend]
  B --> C[Async Worker Thread]
  C --> D[GitHub API Fetch]
  D --> E[Embedding Model MiniLM-L6-v2]
  E --> F[FAISS Vector Index]
  F --> G[Retrieve Top-k Code Files]
  G --> H[GPT-4o Summaries]
  H --> I[Final Combined Summary]
  I --> J[Store JSON]
  J --> K[Dashboard UI]




🔍 Retrieval & Summarization Pipeline


sequenceDiagram
  participant U as User
  participant S as Flask Server
  participant W as Worker Thread
  participant GH as GitHub API
  participant E as Embedding Model
  participant F as FAISS
  participant O as OpenAI GPT-4o

  U->>S: Submit repository URL
  S->>W: Start async job
  W->>GH: Fetch repo contents
  GH-->>W: Code files returned
  W->>E: Generate embeddings
  W->>F: Build FAISS index
  W->>F: Query for relevant snippets
  F-->>W: Top-k code files
  W->>O: Summarize files
  O-->>W: File summaries
  W->>O: Request final summary
  O-->>W: Project-level summary
  W->>S: Save results
  S-->>U: View in dashboard
```

📁 Project Structure
```
Code-Summarize-RAG/
│
├── app.py                     # Flask backend + async job manager
│
├── Summarizer/
│   └── summarizer.py          # Embeddings, FAISS, GPT summarization
│
├── templates/
│   └── index.html             # Dashboard UI
│
├── requirements.txt
└── README.md
```


🧠 Technical Highlights
1. Transformer Embeddings

  - Uses sentence-transformers/all-MiniLM-L6-v2
  - Encodes files into 384-dimensional embeddings

2. FAISS Vector Search

  - L2 nearest-neighbor search
  - Efficient retrieval of relevant code files

3. GPT-4o Summaries

  - File-level summaries
  - High-level overall summary
  - Consistent output via structured prompts

4. Backend Engineering

  - Flask routing
  - Thread-based asynchronous workers
  - Input validation + duplicate detection

5. Frontend Design

  - Searchable table of summaries
  - Clean minimal UI
  - AJAX updates for smooth interaction


📦 Installation
```
git clone https://github.com/yourusername/Code-Summarize-RAG.git
cd Code-Summarize-RAG
pip install -r requirements.txt
```

▶️ Running the App
```
export OPENAI_API_KEY=your_key
export processed_repos=/tmp
python app.py
```

Open in browser:
```
http://localhost:5000
```

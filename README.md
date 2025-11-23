Code-Summarize-RAG

A Retrieval-Augmented Generation (RAG) system that automatically analyzes GitHub repositories, retrieves semantically relevant code files, and generates high-quality AI summaries using OpenAI models, FAISS vector search, and transformer-based embeddings.

This project showcases full-stack AI engineering — combining NLP, vector databases, backend systems, and applied ML — making it suitable for hiring teams and graduate admissions committees evaluating AI/ML competency.

✨ Overview

Code-Summarize-RAG is an end-to-end platform that:

Fetches and analyzes public GitHub repositories

Extracts source code across multiple languages

Embeds files with MiniLM-L6-v2 (Sentence Transformers)

Builds a FAISS index for high-speed semantic retrieval

Uses GPT-4o to produce:

File-level summaries

A final high-level project summary

Stores processed results in /tmp/processed_repos.json

Displays all summaries in a responsive dashboard UI

This is a practical demonstration of RAG techniques applied to code intelligence, highlighting skills in modern AI system design.

🚀 Key Features
🔎 Retrieval-Augmented Generation (RAG)

Embeds all code files using transformer embeddings

Uses FAISS for similarity search

Selects the most relevant files before summarization

Produces more accurate and context-aware summaries

🧠 AI-Driven Summaries

GPT-4o file summaries (1–2 sentences each)

GPT-4o repository-level summary (2–3 sentences)

Identifies technologies, frameworks, and structure

🖥 Full-Stack System

Flask backend for job orchestration

Threaded async processing

HTML/JS UI with search + table view

Clean architecture ready for extension

🎓 Academic + Professional Relevance

Demonstrates engineering capabilities across:

Information retrieval

Embedding models

Intelligent summarization

Cloud-aware backend design

Web application development

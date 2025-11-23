Code-Summarize-RAG

A Retrieval-Augmented Generation (RAG) system that automatically analyzes software repositories, retrieves the most semantically relevant code files, and generates high-quality AI summaries using OpenAI models, FAISS, and modern embedding techniques.

This project demonstrates end-to-end capability across machine learning, software engineering, information retrieval, backend systems, and full-stack deployment. It reflects a practical application of RAG principles to real-world engineering workflows—an increasingly critical capability in modern AI systems.

✨ Overview

Code-Summarize-RAG is a full-stack AI platform that:

Crawls any public GitHub repository

Extracts meaningful source code across multiple languages

Converts code into dense vector embeddings using Sentence-Transformers

Indexes them with FAISS for fast semantic search

Retrieves the most relevant files via similarity search

Uses GPT-4o to produce both file-level and high-level project summaries

Stores results in a lightweight datastore

Displays summaries in a clean, interactive HTML dashboard

This system combines elements of NLP, vector search, distributed information retrieval, and web backend engineering, providing a practical showcase of RAG methodologies applied to code intelligence.

🚀 Key Features
Retrieval-Augmented Code Understanding

Embeds entire repositories using a pretrained MiniLM transformer

Searches semantically for the most relevant code snippets

Summarizes retrieved files with OpenAI models

Full-Stack Architecture

Flask backend for asynchronous job orchestration

Threaded job manager for non-blocking repository processing

Interactive UI for searching & viewing summaries

Persistent JSON storage for processed repos

Academic Relevance

This project demonstrates:

Practical application of RAG pipelines

Building domain-specific embeddings

Using FAISS vector databases

Layering semantic search with LLM-based summarization

Designing end-to-end ML systems from ingestion → retrieval → generation

It shows competence across the pipeline—from systems design to inference optimization—mirroring the applied AI and ML engineering expected in graduate-level work.

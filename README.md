✨ Overview

Code-Summarize-RAG is an end-to-end platform that:

Fetches and analyzes public GitHub repositories

Extracts source code across multiple languages

Embeds code files using MiniLM-L6-v2 (Sentence Transformers)

Builds a FAISS index for high-speed semantic retrieval

Uses GPT-4o to generate:

File-level summaries

A combined high-level project summary

Stores processed results in /tmp/processed_repos.json

Displays summaries in an interactive web dashboard

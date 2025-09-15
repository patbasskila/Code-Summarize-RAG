import os
import re
import json
import base64
import datetime
import requests
import smtplib
import numpy as np
import faiss
import torch

from transformers import AutoTokenizer, AutoModel
from email.mime.text import MIMEText
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Optional GitHub token for private repos
github_token = os.environ.get("GITHUB_TOKEN")

# Path to processed repos JSON
processed_repos = os.environ.get("processed_repos", "/tmp")

# Allow duplicate lib load for FAISS/Torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load embedding model from HuggingFace
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def get_source_repos():
    """Get repository URLs from environment variable or config.json."""
    repo_urls = os.environ.get("SOURCE_REPO_URLS")
    if repo_urls:
        return repo_urls.split(",")

    with open("config.json", "r") as config_file:
        config = json.load(config_file)
        return config.get("source_repo_urls", [])


def get_code_from_github(repo_url):
    """
    Fetch code snippets from a GitHub repository using the GitHub REST API.
    Only retrieves files with specific extensions.
    """
    parts = repo_url.strip().split("/")
    owner, repo = parts[-2], parts[-1]
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"

    headers = {
        "Authorization": f"token {github_token}" if github_token else "",
        "Accept": "application/vnd.github.v3+json",
    }

    def fetch_files(url):
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch repo contents: {response.status_code}")

        items = response.json()
        code_snippets = []
        for item in items:
            if item["type"] == "file" and item["name"].endswith(
                (".py", ".js", ".java", ".sh", ".ps1", ".yaml", ".yml",
                 ".cs", ".sql", ".ts", ".cshtml")
            ):
                file_response = requests.get(item["download_url"], headers=headers)
                if file_response.status_code == 200:
                    code_snippets.append(file_response.text)
                else:
                    print(f"Skipping file {item['name']} due to fetch error.")
            elif item["type"] == "dir":
                subdir_snippets = fetch_files(item["url"])
                code_snippets.extend(subdir_snippets)
        return code_snippets

    all_code_snippets = fetch_files(api_url)
    if not all_code_snippets:
        raise Exception("No valid code files found in the repository.")
    return all_code_snippets


def embed_text(text):
    """Generate embeddings for the given text using the loaded model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()
    return embeddings


def create_faiss_index(codes):
    """Create a FAISS index for fast similarity search on code embeddings."""
    dimension = 384
    index = faiss.IndexFlatL2(dimension)
    embeddings = [embed_text(code) for code in codes]
    embeddings = np.vstack(embeddings)
    index.add(embeddings)
    return index, embeddings


def search_faiss_index(index, query_embedding, embeddings, codes, k=3):
    """Search FAISS index for top-k relevant code snippets."""
    D, I = index.search(query_embedding, k)
    return [codes[i] for i in I[0]]


def summarize_code_snippets(file_contents):
    """Use OpenAI API to generate summaries for individual code files."""
    summaries = []
    for file_name, code in file_contents.items():
        prompt = f"""
        Analyze and summarize the following code file. Provide a concise summary (1–2 sentences)
        covering purpose, key components, and functionality.
        Highlight languages, frameworks, or technologies (Python, Ansible, Kubernetes, etc.)
        using HTML <b> tags only.

        File: {file_name}
        Code:
        {code}
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert coding assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=2000,
            )
            summaries.append(
                f"{file_name}: {response.choices[0].message.content.strip()}"
            )
        except Exception as e:
            summaries.append(f"{file_name}: Error generating summary: {str(e)}")
    return summaries


def generate_final_summary(summaries):
    """Use OpenAI API to generate a high-level overview summary."""
    combined_prompt = f"""
    Below are summaries of individual files from a repository:

    {summaries}

    Write a concise 2–3 sentence high-level overview of the project.
    Mention languages, frameworks, or technologies using HTML <b> tags only.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert software analyst."},
                {"role": "user", "content": combined_prompt},
            ],
            temperature=0,
            max_tokens=3000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating final summary: {str(e)}"


def process_repository(repo_url):
    """
    Main function to process a repository:
    - Fetch code from GitHub
    - Create FAISS index
    - Retrieve relevant snippets
    - Generate summaries
    """
    print(f"Processing repo: {repo_url}")
    code_files = get_code_from_github(repo_url)
    index, embeddings = create_faiss_index(code_files)
    query_embedding = embed_text("\n".join(code_files))
    relevant_codes = search_faiss_index(index, query_embedding, embeddings, code_files)
    individual_summaries = summarize_code_snippets(
        {f"file_{i}": code for i, code in enumerate(relevant_codes)}
    )
    final_summary = generate_final_summary("\n".join(individual_summaries))
    return final_summary


def load_processed_repos(file_path=processed_repos + "/processed_repos.json"):
    """Load processed repositories from JSON file."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            return data.get("processed", [])
    except FileNotFoundError:
        return []


def update_processed_repos(repo_url, summary, file_path=processed_repos + "/processed_repos.json"):
    """Append a new processed repository entry to the JSON file."""
    processed = load_processed_repos(file_path)
    timestamp = datetime.datetime.now().strftime("%m/%d/%Y")
    new_entry = {"repo_url": repo_url, "summary": summary, "timestamp": timestamp}
    processed.append(new_entry)
    with open(file_path, "w") as f:
        json.dump({"processed": processed}, f)


def send_email_notification(recipient_email, summary, error_message):
    """Send an email notification with the processing outcome."""
    from_email = "noreply@example.com"
    subject = "Repository Processing Outcome"

    if error_message:
        body = f"Processing failed with error:\n\n{error_message}"
    else:
        body = f"Processing completed successfully. Summary:\n\n{summary}"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = recipient_email

    try:
        with smtplib.SMTP("localhost") as server:
            server.send_message(msg)
        print(f"Email sent to {recipient_email}")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")

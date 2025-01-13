import requests
import os
import openai
from openai import OpenAI
from github import Github
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Initialize OpenAI API key
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Initialize GitHub API token
github_token = os.environ.get("GITHUB_TOKEN")
g = Github(github_token)

# Initialize the model and tokenizer for generating embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_code_from_github(repo_url):
    parts = repo_url.split('/')
    owner, repo = parts[-2], parts[-1]
    repository = g.get_repo(f"{owner}/{repo}")
    
    # Use a default file if "pull.py" doesn't exist
    try:
        content_file = repository.get_contents("pull.py")
    except:
        content_file = repository.get_contents("README.md")
    
    return content_file.decoded_content.decode('utf-8')

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()
    return embeddings

def create_faiss_index(codes):
    dimension = 384  # Embedding size for the selected model
    index = faiss.IndexFlatL2(dimension)
    embeddings = [embed_text(code) for code in codes]
    embeddings = np.vstack(embeddings)
    index.add(embeddings)
    return index, embeddings

def search_faiss_index(index, query_embedding, embeddings, codes, k=3):
    D, I = index.search(query_embedding, k)
    return [codes[i] for i in I[0]]

def generate_code_summary(codes):
    combined_code = "\n".join(codes)
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a coding assistant. Summarize the given code briefly and accurately."},
                {"role": "user", "content": f"Summarize the following code in one or two sentences:\n\n{combined_code}"}
            ],
            temperature=0.5,
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def update_markdown_with_summary(repo_url, summary):
    repo_name = "patbasskila/Pull-latest-Models"
    file_path = "summaries.md"
    
    repo = g.get_repo(repo_name)
    file = repo.get_contents(file_path)
    
    updated_content = file.decoded_content.decode('utf-8') + f"\n- [{repo_url}]({repo_url}): {summary}"
    repo.update_file(file.path, "Added new code summary", updated_content, file.sha)

def main():
    repo_url = "https://github.com/patbasskila/Pull-latest-Models"
    
    code = get_code_from_github(repo_url)
    code_snippets = code.split('\n\n')
    
    index, embeddings = create_faiss_index(code_snippets)
    query_embedding = embed_text(code)
    relevant_codes = search_faiss_index(index, query_embedding, embeddings, code_snippets)
    summary = generate_code_summary(relevant_codes)
    print(summary)
    
    update_markdown_with_summary(repo_url, summary)
    print("Summary added successfully!")

if __name__ == "__main__":
    main()

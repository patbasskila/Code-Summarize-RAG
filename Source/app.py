from flask import Flask, render_template, request, jsonify
import threading
import time
import json
import re
import os

from Summarizer.summarizer import (
    process_repository,
    update_processed_repos,
    load_processed_repos,
    send_email_notification,
)

app = Flask(__name__)
processed_repos = os.environ.get("processed_repos")

# Global dictionary to store job statuses for asynchronous processing
# Format: {job_id: {"status": "queued"/"processing"/"completed"/"failed", "message": "..."}}
job_statuses = {}


def background_job(job_id, repo_url, email):
    """
    Background job that processes a repository.
    Updates the job status, calls the summarization function,
    updates the /tmp/processed_repos.json file, and sends an email notification.
    """
    job_statuses[job_id]["status"] = "processing"
    try:
        summary = process_repository(repo_url)

        # Update the /tmp/processed_repos.json file with new summary info
        update_processed_repos(repo_url, summary)

        job_statuses[job_id]["status"] = "completed"
        job_statuses[job_id]["message"] = summary

        # Send success email
        send_email_notification(email, summary, None)
    except Exception as e:
        error_msg = str(e)
        job_statuses[job_id]["status"] = "failed"
        job_statuses[job_id]["message"] = error_msg

        # Send failure email
        send_email_notification(email, None, error_msg)


@app.route("/")
def index():
    """
    Render the main index page.
    Loads summaries from /tmp/processed_repos.json and passes them to the template.
    """
    try:
        with open(processed_repos + "/processed_repos.json", "r") as f:
            data = json.load(f)
    except Exception:
        data = {"processed": []}
    return render_template("index.html", summaries=data["processed"])


@app.route("/submit-repo", methods=["POST"])
def submit_repo():
    """
    Endpoint to receive new repository submissions via AJAX.
    Performs server-side validation for repository URL and email (must end with @example.com).
    Launches a background thread to process the repository.
    """
    data = request.get_json()
    repo_url = data.get("repo_url", "").strip()
    email = data.get("email", "").strip()

    if not repo_url.startswith("http"):
        return jsonify({"error": "Invalid repository URL"}), 400

    if repo_url.endswith(".git"):
        repo_url = repo_url[:-4]

    # Email must end with "@example.com"
    if not re.match(r".+@example\.com$", email):
        return jsonify({"error": "Email must be a valid @example.com address"}), 400

    processed = load_processed_repos()
    if any(entry.get("repo_url") == repo_url for entry in processed):
        return jsonify({"error": "This repository has already been processed."}), 400

    job_id = str(int(time.time() * 1000))
    job_statuses[job_id] = {"status": "queued", "message": "Job is queued for processing."}

    thread = threading.Thread(target=background_job, args=(job_id, repo_url, email))
    thread.start()

    return jsonify({"job_id": job_id, "message": "Job has been queued."}), 200


@app.route("/job-status/<job_id>", methods=["GET"])
def job_status(job_id):
    """
    Endpoint for AJAX polling.
    Returns the status of a submitted job given its job_id.
    """
    status = job_statuses.get(job_id, None)
    if status:
        return jsonify(status)
    else:
        return jsonify({"error": "Job not found"}), 404


@app.route("/all-summaries", methods=["GET"])
def all_summaries():
    """
    Endpoint to return all processed summaries (used to refresh the frontend table).
    """
    try:
        with open(processed_repos + "/processed_repos.json", "r") as f:
            data = json.load(f)
    except Exception:
        data = {"processed": []}
    return jsonify(data["processed"])


if __name__ == "__main__":
    # Run Flask app on all interfaces (0.0.0.0) on port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)

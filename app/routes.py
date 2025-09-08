from flask import Blueprint, render_template, request, send_file
from .utils import rank_resumes, extract_text_from_pdf, extract_email
import pandas as pd
from pathlib import Path
import os
from scripts.SBERT.resume_ranker_sbert import run_resume_ranker
import time

main = Blueprint("main", __name__)
RESULTS_DIR = Path("/tmp/Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

@main.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        job_desc_text_input = request.form.get("job_desc_text", "").strip()
        job_desc_file = request.files.get("job_desc_file")
        resume_files = request.files.getlist("resume_files")
        keywords = request.form.get("keywords")
        top_n = int(request.form.get("top_n", 5))

        if len(resume_files) > 50:
            return render_template("index.html", error="Maximum 50 resumes allowed.")

        job_desc_text = ""

        # Extract text if file uploaded
        if job_desc_file and job_desc_file.filename:
            job_desc_text = extract_text_from_pdf(job_desc_file).strip()

        # Use manual input if provided (overrides file)
        if job_desc_text_input:
            job_desc_text = job_desc_text_input

        # Check if at least one JD input provided
        if not job_desc_text:
            return render_template("index.html", error="Please provide a job description (text or file).")

        valid_resumes = [f for f in resume_files if f.filename.endswith((".txt", ".pdf"))]
        top_n = min(top_n, len(valid_resumes))

        if not valid_resumes:
            return render_template("index.html", error="Please upload at least one valid .txt or .pdf resume.")

        # Build resumes DataFrame with emails
        data = []
        for f in resume_files:
            if f.filename.endswith(".txt"):
                text = f.read().decode("utf-8", errors="ignore").strip() or "No content"
            elif f.filename.endswith(".pdf"):
                text = extract_text_from_pdf(f)
            else:
                continue
            email = extract_email(text)
            data.append({"ID": f.filename, "Resume_str": text, "email": email})

        resumes_df = pd.DataFrame(data)

        # Run SBERT ranker
        results = run_resume_ranker(
            resumes_df,
            job_desc_text,
            top_n=top_n,
            keywords=keywords.split(",") if keywords else None
        )

        if not results or "custom" not in results:
            return render_template("index.html", error="No matching resumes found.")

        top_resumes = results["custom"]

        # Ensure email column exists
        if 'email' not in top_resumes.columns:
            top_resumes['email'] = top_resumes['ID'].map(resumes_df.set_index('ID')['email'])

        csv_path = RESULTS_DIR / f"results_{int(time.time())}.csv"
        top_resumes.to_csv(csv_path, index=False)

        return render_template(
            "results.html",
            top_resumes=top_resumes.to_dict(orient="records"),
            job_desc=job_desc_text,
            csv_path=csv_path
        )

    return render_template("index.html")

@main.route("/download/<path:filename>")
def download_file(filename):
    return send_file(RESULTS_DIR / filename, as_attachment=True)
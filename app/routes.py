from flask import Blueprint, render_template, request, send_file
from .utils import extract_text_from_pdf, extract_email, rank_resumes
import pandas as pd
from pathlib import Path
import os
import time

main = Blueprint("main", __name__)
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

@main.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        job_desc_text_input = request.form.get("job_desc_text", "").strip()
        job_desc_file = request.files.get("job_desc_file")
        resume_files = request.files.getlist("resume_files")
        keywords = request.form.get("keywords", "").strip()
        top_n = int(request.form.get("top_n", 5))

        job_desc_text = ""

        if job_desc_file and job_desc_file.filename:
            # We assume JD files are TXT based on index.html
            job_desc_text = job_desc_file.read().decode("utf-8", errors="ignore").strip()

        if job_desc_text_input:
            job_desc_text = job_desc_text_input

        if not job_desc_text:
            return render_template("index.html", error="Please provide a job description (text or .txt file).")

        valid_resumes = [f for f in resume_files if f.filename and f.filename.lower().endswith((".txt", ".pdf"))]

        if not valid_resumes:
            return render_template("index.html", error="Please upload at least one valid .txt or .pdf resume.")
        
        # Pass the uploaded files directly to the utility function for processing
        top_resumes_df, error = rank_resumes(
            job_desc_text=job_desc_text,
            keywords=keywords,
            top_n=top_n,
            uploaded_resumes=valid_resumes
        )

        if error:
            return render_template("index.html", error=error)

        if top_resumes_df.empty:
            return render_template("index.html", error="No matching resumes found after keyword filtering or ranking.")

        # Save results to CSV
        csv_path = RESULTS_DIR / f"results_{int(time.time())}.csv"
        # Exporting only the columns visible to the user
        top_resumes_df[['ID', 'similarity', 'email', 'key_matches']].to_csv(csv_path, index=False)

        return render_template(
            "results.html",
            top_resumes=top_resumes_df.to_dict(orient="records"),
            job_desc=job_desc_text,
            csv_filename=csv_path.name
        )

    return render_template("index.html")

@main.route("/download/<filename>")
def download_file(filename):
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        return f"File not found: {filename}", 404
    # The MIME type is now text/csv, not application/octet-stream (more correct)
    return send_file(file_path, as_attachment=True, mimetype='text/csv')

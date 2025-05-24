from flask import Blueprint, render_template, request, send_file
from .utils import rank_resumes
import pandas as pd
from pathlib import Path
import os
from scripts.SBERT.resume_ranker_sbert import run_resume_ranker
import pandas as pd
from PyPDF2 import PdfReader
import time

main = Blueprint("main", __name__)

# Get base path from environment variable
# BASE_PATH = Path(os.getenv('BASE_PATH', os.path.dirname(os.path.abspath(__file__))))
# RESULTS_DIR = BASE_PATH / "Results"
RESULTS_DIR = Path("/tmp/Results")

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

@main.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        job_desc_text = request.form.get("job_desc_text")
        job_desc_file = request.files.get("job_desc_file")
        resume_files = request.files.getlist("resume_files")
        keywords = request.form.get("keywords")
        top_n = int(request.form.get("top_n", 5))
        
        # Limit number of uploaded resumes
        if len(resume_files) > 50:
            return render_template("index.html", error="Maximum 50 resumes allowed.")
        
        # Process job description
        if job_desc_file and job_desc_file.filename:
            job_desc_text = job_desc_file.read().decode("utf-8", errors="ignore").strip() or "No content"
        
        if not job_desc_text:
            return render_template("index.html", error="Please provide a job description.")
        
        # Cap top_n at number of valid resumes
        valid_resumes = [f for f in resume_files if f.filename.endswith((".txt", ".pdf"))]
        top_n = min(top_n, len(valid_resumes))
        
        if not valid_resumes:
            return render_template("index.html", error="Please upload at least one valid .txt or .pdf resume.")
        
        # Rank resumes
        # top_resumes, error = rank_resumes(job_desc_text, keywords, top_n, uploaded_resumes=resume_files)
        
        # if error:
        #     return render_template("index.html", error=error)

        # Convert uploaded files into DataFrame
        data = []
        for f in resume_files:
            if f.filename.endswith(".txt"):
                text = f.read().decode("utf-8", errors="ignore")
            elif f.filename.endswith(".pdf"):
                reader = PdfReader(f)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            else:
                continue
            data.append({"ID": f.filename, "Resume_str": text})

        resumes_df = pd.DataFrame(data)

        # Run the SBERT ranker
        results = run_resume_ranker(resumes_df, job_desc_text, top_n=top_n, keywords=keywords.split(",") if keywords else None)

        if not results or "custom" not in results:
            return render_template("index.html", error="No matching resumes found.")

        top_resumes = results["custom"]

        
        # Save results to CSV
        # BASE_PATH = Path("/home/madhukiran/Desktop/Elevate Labs/Project/ai-resume-ranker")
        # csv_path = BASE_PATH / "Results/results.csv"
        # top_resumes.to_csv(csv_path, index=False)
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
    # BASE_PATH = Path("/home/madhukiran/Desktop/Elevate Labs/Project/ai-resume-ranker")
    # return send_file(BASE_PATH / filename, as_attachment=True)
    return send_file(RESULTS_DIR / filename, as_attachment=True)
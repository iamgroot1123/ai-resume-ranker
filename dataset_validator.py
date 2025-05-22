import os
import pandas as pd
from pathlib import Path
import pdfplumber
from tqdm import tqdm
import logging

# Suppress CropBox warnings
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

def normalize_category(category):
    """Normalize category names (e.g., 'HR' -> 'hr', 'INFORMATION-TECHNOLOGY' -> 'information-technology')."""
    return category.lower().replace("-", " ")

def check_folder_structure(resumes_path):
    """Check if resumes/ folder has required subfolders."""
    required_folders = ["data", "job_descriptions"]
    print("Checking folder structure...")
    missing_folders = []
    for folder in required_folders:
        if not (resumes_path / folder).exists():
            missing_folders.append(folder)
    if missing_folders:
        print(f"Warning: Missing folders: {missing_folders}")
    else:
        print("All required folders found.")
    
    # Check category folders in data/
    data_path = resumes_path / "data"
    if data_path.exists():
        categories = [d.name for d in data_path.iterdir() if d.is_dir()]
        print(f"Found {len(categories)} category folders in data/: {categories}")

def count_files(folder_path, extension):
    """Count files with given extension in folder and subfolders."""
    return sum(1 for file in folder_path.rglob(f"*.{extension}"))

def validate_pdfs(data_path, csv_data, max_pdfs=20):
    """Check if PDFs exist and are text-extractable (limited to max_pdfs)."""
    print(f"\nValidating up to {max_pdfs} PDFs...")
    pdf_ids = set(csv_data["ID"].astype(str) + ".pdf")
    sample_ids = list(pdf_ids)[:max_pdfs]
    missing_pdfs = []
    non_extractable = []
    
    for pdf_id in tqdm(sample_ids, desc="Validating PDFs"):
        found = False
        for category_path in data_path.iterdir():
            if (category_path / pdf_id).exists():
                found = True
                try:
                    with pdfplumber.open(category_path / pdf_id) as pdf:
                        text = pdf.pages[0].extract_text() or ""
                        if not text.strip():
                            non_extractable.append(str(category_path / pdf_id))
                        else:
                            print(f"OK: {category_path / pdf_id} is text-extractable.")
                except Exception as e:
                    print(f"Error: {category_path / pdf_id} failed to open: {e}")
                break
        if not found:
            missing_pdfs.append(pdf_id)
    
    if missing_pdfs:
        print(f"Warning: {len(missing_pdfs)} PDFs listed in Resume.csv not found: {missing_pdfs[:5]}...")
    if non_extractable:
        print(f"Warning: {len(non_extractable)} PDFs are not text-extractable: {non_extractable[:5]}...")

def summarize_csv(csv_path, job_desc_path):
    """Summarize Resume.csv and check job descriptions."""
    print("\nSummarizing Resume.csv...")
    if not csv_path.exists():
        print("Error: Resume.csv not found.")
        return None
    try:
        df = pd.read_csv(csv_path)
        print(f"Total resumes: {len(df)}")
        categories = df["Category"].unique().tolist()
        print(f"Categories in Resume.csv: {categories}")
        print("Resumes per category:")
        print(df["Category"].value_counts().to_string())
        missing = df.isnull().sum()
        if missing.any():
            print("Warning: Missing values in Resume.csv:", missing[missing > 0].to_dict())
        
        # Check job descriptions against categories
        job_desc_files = [f.stem for f in job_desc_path.glob("*.txt")]
        normalized_csv_categories = [normalize_category(c) for c in categories]
        normalized_job_desc_categories = [normalize_category(c) for c in job_desc_files]
        missing_job_desc = [c for c in categories if normalize_category(c) not in normalized_job_desc_categories]
        extra_job_desc = [c for c in job_desc_files if normalize_category(c) not in normalized_csv_categories]
        
        if missing_job_desc:
            print(f"Warning: Job descriptions missing for categories: {missing_job_desc}")
        if extra_job_desc:
            print(f"Note: Extra job descriptions not in Resume.csv: {extra_job_desc}")
        
        return df
    except Exception as e:
        print(f"Error reading Resume.csv: {e}")
        return None

def validate_dataset(resumes_path):
    """Validate and summarize the entire dataset."""
    resumes_path = Path(resumes_path).resolve()
    
    # Check folder structure
    check_folder_structure(resumes_path)
    
    # Count resumes and job descriptions
    pdf_count = count_files(resumes_path / "data", "pdf")
    txt_count = count_files(resumes_path / "job_descriptions", "txt")
    print(f"\nDataset summary:")
    print(f"Total resumes (PDFs): {pdf_count}")
    print(f"Total job descriptions (TXT): {txt_count}")
    
    # Summarize Resume.csv and check job descriptions
    csv_path = resumes_path / "Resume.csv"
    job_desc_path = resumes_path / "job_descriptions"
    csv_data = summarize_csv(csv_path, job_desc_path)
    
    # Validate PDFs (sample up to 20)
    if csv_data is not None:
        sample_data = csv_data.sample(n=min(20, len(csv_data)), random_state=42)
        validate_pdfs(resumes_path / "data", sample_data, max_pdfs=20)

if __name__ == "__main__":
    # Your resumes/ folder path
    RESUMES_PATH = "/home/madhukiran/Desktop/Elevate Labs/Project/ai-resume-ranker/resumes"
    
    if not os.path.exists(RESUMES_PATH):
        print(f"Error: Folder {RESUMES_PATH} does not exist. Please check path.")
    else:
        validate_dataset(RESUMES_PATH)

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
import pdfplumber
import re
import nltk
from nltk.corpus import stopwords
from flask import current_app
from typing import Dict, Any, List
import sqlite3
from pathlib import Path
import time

nltk.download('stopwords', quiet=True)

# --- Database Initialization (Placeholder, will be integrated next) ---
DATABASE_PATH = Path(__file__).resolve().parent.parent / "Results" / "parsed_resumes.db"
def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parsed_resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            email TEXT,
            skills TEXT,
            experience TEXT,
            education TEXT,
            raw_text TEXT,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# IMPORTANT: Call this initialization function once when the module loads
init_db() 
# --- End DB Placeholder ---


# --- LLM/SBERT Model Loading (Existing) ---
def load_model_once():
    """Load SBERT model once upon application startup."""
    try:
        # Note: Keeping the original model for now as requested
        model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
        print("SBERT model loaded successfully for application context.")
        return model
    except Exception as e:
        print(f"Error loading SBERT model: {e}")
        return None

# --- PDF/Email Extraction (Existing) ---
def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text.strip() or "No content"
    except Exception as e:
        try:
            file.seek(0)
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
            return text.strip() or "No content"
        except:
            return "No content"

def extract_email(text):
    if not isinstance(text, str) or not text.strip():
        return "No email found"
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    match = re.search(email_pattern, text)
    return match.group(0) if match else "No email found"

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()
    text = re.sub(r'\b(professional summary|summary|references|education|certifications)\b.*?(?=\b\w+\b|$)', '', text, flags=re.IGNORECASE)
    return text or "No content"

# --- Structured Data Extraction --

def extract_section_text(text: str, section_name: str, next_section_names: List[str]) -> str:
    """Extracts text content for a specific section (e.g., 'Education') using a pattern that prioritizes stop words."""
    
    section_name_normalized = section_name.strip()
    
    # 1. Create the START pattern: Find the header, allowing for any surrounding whitespace/symbols on a line.
    # We use a pattern that captures the header and the text that follows it.
    pattern_start = r'[\r\n]{1,3}\s*[\-\*\.]*\s*' + re.escape(section_name_normalized) + r'\s*[:\-\.]{0,5}\s*[\r\n]{1,3}'
    
    # 2. Create the STOP pattern: Match the next header on a line by itself.
    # FIX: Make the stop pattern very simple (just the header name surrounded by flexible whitespace)
    # and use it as the main lookahead boundary.
    stop_patterns = [r'\s*' + re.escape(name.strip()) + r'\s*' for name in next_section_names if name.strip() != section_name_normalized]
    pattern_stop = r'|'.join(stop_patterns)
    
    # 3. Final regex: Find the START, then capture content NON-GREEDILY (.*?) until the STOP pattern or end of string.
    # We remove the line break from the stop pattern regex itself, relying on the lookahead assertion (.*?)
    # to stop immediately before the next header.
    regex = re.compile(
        pattern_start + r'(.*?)(?=\s*(' + pattern_stop + r')|$)', # Lookahead for the next header (or end of string)
        re.IGNORECASE | re.DOTALL | re.MULTILINE
    )
    
    # Normalize newlines and symbols in the input text once for consistency
    text_normalized = re.sub(r'[\r\n]+', '\n', text)
    # Aggressively remove common symbols used as bullets that may interfere with capture
    text_normalized = re.sub(r'[\-\*\•]', '', text_normalized) 
    
    match = regex.search(text_normalized)

    if match:
        content = match.group(1).strip()
        # Clean up excessive newlines/spaces within the content
        return re.sub(r'[\r\n]{2,}', '\n', content).strip()
    
    return "Not Found"


def parse_resume_sections(text: str) -> Dict[str, str]:
    """Extracts Skills, Experience, and Education from raw resume text."""
    
    # Re-examine ALL_SECTIONS list to ensure all major headers are present
    ALL_SECTIONS = [
        "SKILLS", "TECHNICAL SKILLS", "EXPERIENCE", "WORK HISTORY", "PROFESSIONAL EXPERIENCE", 
        "EDUCATION", "PROJECTS", "CERTIFICATION", "ACHIEVEMENTS", "SUMMARY", "CONTACT", "REFERENCES"
    ]
    
    # --- 1. Extract Education ---
    # Try common headers for education
    education_text = extract_section_text(text, "EDUCATION", ALL_SECTIONS)
    
    # Education pattern: Look for degree/university info followed by a year
    education_pattern = r'((?:b\.\s?tech|m\.\s?s|ph\.\s?d|bachelor|master|degree|diploma|college|university)[\s\w\d,&-]*?\d{4})'
    education_mentions = [match.strip() for match in re.findall(education_pattern, education_text, re.IGNORECASE)]
    
    # --- 2. Extract Experience (Work History) ---
    experience_text = extract_section_text(text, "PROFESSIONAL EXPERIENCE", ALL_SECTIONS)
    if experience_text == "Not Found":
        experience_text = extract_section_text(text, "WORK EXPERIENCE", ALL_SECTIONS)
    if experience_text == "Not Found":
        experience_text = extract_section_text(text, "EXPERIENCE", ALL_SECTIONS)
    
    # Experience pattern: Simple match for Job Title + Role Type + Company Separator
    experience_pattern = r'(\w+\s+(engineer|developer|analyst|scientist|manager|consultant|intern)[\s\w\d,-]*?(at|on|for|\|)\s*[\w\s\&-]+)'
    experience_mentions = [match[0].strip() for match in re.findall(experience_pattern, experience_text, re.IGNORECASE)]
    
    # --- 3. Extract Skills (UNIVERSAL & DEEPER EXTRACTION) ---
    
    skills_text = extract_section_text(text, "SKILLS", ALL_SECTIONS)
    if skills_text == "Not Found":
        skills_text = extract_section_text(text, "TECHNICAL SKILLS", ALL_SECTIONS)
    
    # Universal Fix: Remove internal sub-headers to fuse the skill list into one continuous block.
    # FIX: Simplify the sub-header pattern as we removed bullet points earlier.
    SUB_HEADER_PATTERN = r'^\s*(Languages|Libraries/Frameworks|Others|Core Libraries|ML/DL Frameworks|Data Libraries|Cloud & DevOps|Tools|Concepts|Programming Languages|Core|Deep Learning|Technical)\s*[:\-\.]{0,2}\s*[\r\n]'
    
    cleaned_skills_text = re.sub(SUB_HEADER_PATTERN, ' ', skills_text, flags=re.IGNORECASE | re.MULTILINE).strip()
    
    # Final cleanup: replace multiple newlines/spaces with a single separator for display
    universal_skills = re.sub(r'[\r\n]+', ', ', cleaned_skills_text).strip()
    
    # Fallback (optional) - Removed for now to ensure section logic is fully verified.
    
    return {
        "education": " | ".join(education_mentions) or "Not specified",
        "experience": " | ".join(experience_mentions) or "Not specified",
        "skills": universal_skills if universal_skills and universal_skills != "Not Found" else "Not specified"
    }


# --- Ranking and Similarity Logic (Existing/Needs Future Refactoring) ---

def vectorize_texts_sbert(texts, model, batch_size=100):
    # ... existing implementation ...
    valid_texts = [clean_text(t) if isinstance(t, str) and t.strip() else "No content" for t in texts]
    vectors = []
    for i in range(0, len(valid_texts), batch_size):
        batch = valid_texts[i:i + batch_size]
        batch_vectors = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        vectors.append(batch_vectors)
    return np.vstack(vectors)

def extract_keywords_from_job_desc(job_desc_text, top_n=10):
    # ... existing implementation ...
    stop_words = set(stopwords.words('english')).union({'seeking', 'expertise', 'summary', 'responsibilities', 'qualifications', 'must', 'have', 'experience', 'ability'})
    vectorizer = TfidfVectorizer(
        stop_words=list(stop_words),
        token_pattern=r'\b\w+\b',
        max_features=50,
        ngram_range=(1, 2)
    )
    try:
        tfidf_matrix = vectorizer.fit_transform([clean_text(job_desc_text)])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray().flatten()
        keyword_scores = [(feature_names[i], tfidf_scores[i]) for i in range(len(feature_names))]
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        keywords = [kw for kw, score in keyword_scores[:top_n] if score > 0]
        return keywords if keywords else ["skills", "project"]
    except:
        return ["skills", "project"]

def filter_resumes(df, keywords, job_desc_text):
    # ... existing implementation ...
    if isinstance(keywords, str) and keywords.strip():
        keywords_list = [k.strip().lower() for k in keywords.split(",")]
    else:
        keywords_list = extract_keywords_from_job_desc(job_desc_text)
    
    filtered_df = df[df["Resume_str"].str.lower().str.contains("|".join(keywords_list), na=False)].copy()
    return filtered_df if not filtered_df.empty else df

def extract_key_matches(resume_text, job_desc_text, top_n=3):
    # ... existing implementation ...
    resume_text = clean_text(resume_text)
    job_desc_text = clean_text(job_desc_text)
    texts = [job_desc_text, resume_text]
    stop_words = set(stopwords.words('english')).union({'seeking', 'expertise', 'summary', 'responsibilities', 'qualifications', 'must', 'have', 'experience', 'ability'})
    
    vectorizer = TfidfVectorizer(
        stop_words=list(stop_words),
        token_pattern=r'\b\w{2,}\b',
        max_features=100,
        ngram_range=(1, 2)
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
    except:
        return ["No significant matches"]
    
    job_tfidf = tfidf_matrix[0].toarray().flatten()
    resume_tfidf = tfidf_matrix[1].toarray().flatten()
    
    common_terms = []
    for i, term in enumerate(feature_names):
        if job_tfidf[i] > 0.05 and resume_tfidf[i] > 0.05:
            common_terms.append((term, job_tfidf[i] + resume_tfidf[i]))
    
    common_terms.sort(key=lambda x: x[1], reverse=True)
    matches = [term for term, _ in common_terms][:top_n]
    
    return [match.replace(" ", "_") for match in matches] if matches else ["No significant matches"]


def rank_resumes(job_desc_text, keywords, top_n, uploaded_resumes):
    """Rank resumes against a job description using a pre-loaded SBERT model and extracts structured data."""
    
    model = current_app.model
    if not model:
        return None, "SBERT model not loaded. Please restart the application."

    job_desc_text = clean_text(job_desc_text.strip() or "No content")
    job_vector = vectorize_texts_sbert([job_desc_text], model)[0]
    
    data = []
    
    for resume in uploaded_resumes:
        filename = resume.filename
        try:
            resume.seek(0) 
            if filename.lower().endswith(".txt"):
                text = resume.read().decode("utf-8", errors="ignore").strip() or "No content"
            elif filename.lower().endswith(".pdf"):
                text = extract_text_from_pdf(resume)
            else:
                continue
            email = extract_email(text)
            
            # --- NEW: Parse Structured Data ---
            parsed_info = parse_resume_sections(text)
            # --- END NEW ---
            
            data.append({
                "ID": filename,
                "Resume_str": text,
                "email": email,
                "parsed_data": parsed_info # Include parsed data for later merging
            })
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue

    if not data:
        return None, "No valid resumes uploaded or processed."
    
    df = pd.DataFrame(data)
    
    # 1. Filter resumes by keywords
    df_filtered = filter_resumes(df, keywords, job_desc_text)
    
    # 2. Cap top_n at available resumes (after filtering)
    top_n = min(top_n, len(df_filtered))
    
    # 3. Calculate SBERT similarity for filtered resumes
    resume_vectors = vectorize_texts_sbert(df_filtered["Resume_str"].tolist(), model)
    similarities = cosine_similarity(resume_vectors, job_vector.reshape(1, -1)).flatten()
    df_filtered.loc[:, "similarity"] = similarities
    
    # 4. Get top results
    top_resumes = df_filtered.sort_values(by="similarity", ascending=False).head(top_n).copy()
    
    # 5. Extract structured data, justification, and calculate rating
    final_top_resumes = []
    
    for index, row in top_resumes.iterrows():
        # Retrieve the original entry to get the full raw text and parsed data
        original_entry = df[df["ID"] == row["ID"]].iloc[0]
        original_resume_text = original_entry["Resume_str"]
        parsed_info = original_entry["parsed_data"]

        matches = extract_key_matches(original_resume_text, job_desc_text)
        
        # LLM Guidance: Convert similarity (0-1) to 1-10 Rating
        # FIX: Replace .round(1) with round(..., 1) for the native float object
        rating_10 = 1 + round(row["similarity"] * 9, 1)
        justification = f"Strong alignment on key terms like: {', '.join(matches).replace('_', ' ')}."

        final_top_resumes.append({
            "ID": row["ID"],
            "similarity": row["similarity"],
            "email": row["email"],
            "rating_10": rating_10,
            "key_matches": ", ".join(matches),
            "justification": justification,
            "skills": parsed_info["skills"],
            "experience": parsed_info["experience"],
            "education": parsed_info["education"],
        })

    return pd.DataFrame(final_top_resumes), None
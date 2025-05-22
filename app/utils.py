import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import pickle
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

def load_model():
    """Load SBERT model."""
    model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
    return model

def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip() or "No content"
    except:
        return "No content"

def extract_email(text):
    """Extract the first email address from text using regex."""
    if not isinstance(text, str) or not text.strip():
        return "No email found"
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    match = re.search(email_pattern, text)
    return match.group(0) if match else "No email found"

def clean_text(text):
    """Clean text by removing noise and normalizing, focusing on skills."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()
    text = re.sub(r'\b(professional summary|summary|references|education|certifications)\b.*?(?=\b\w+\b|$)', '', text, flags=re.IGNORECASE)
    return text or "No content"

def extract_keywords_from_job_desc(job_desc_text, top_n=10):
    """Extract top keywords from job description using TF-IDF."""
    stop_words = set(stopwords.words('english')).union({'seeking', 'expertise', 'summary', 'responsibilities', 'qualifications'})
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
        keywords = [kw for kw, score in keyword_scores[:top_n]]
        return keywords if keywords else ["python", "skills"]
    except:
        return ["python", "skills"]

def filter_resumes(df, keywords, job_desc_text):
    """Filter resumes by keywords extracted from job description or provided."""
    if not keywords:
        keywords = extract_keywords_from_job_desc(job_desc_text)
    else:
        keywords = [k.strip().lower() for k in keywords.split(",")]
    
    filtered_df = df[df["Resume_str"].str.lower().str.contains("|".join(keywords), na=False)].copy()
    return filtered_df if not filtered_df.empty else df

def vectorize_texts_sbert(texts, model, batch_size=100):
    """Vectorize texts using Sentence-BERT."""
    valid_texts = [clean_text(t) if isinstance(t, str) and t.strip() else "No content" for t in texts]
    vectors = []
    for i in range(0, len(valid_texts), batch_size):
        batch = valid_texts[i:i + batch_size]
        batch_vectors = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        vectors.append(batch_vectors)
    return np.vstack(vectors)

def extract_key_matches(resume_text, job_desc_text, top_n=3):
    """Extract key matching phrases using TF-IDF and stopword removal."""
    resume_text = clean_text(resume_text)
    job_desc_text = clean_text(job_desc_text)
    texts = [job_desc_text, resume_text]
    stop_words = set(stopwords.words('english')).union({'seeking', 'expertise', 'summary', 'responsibilities', 'qualifications'})
    
    vectorizer = TfidfVectorizer(
        stop_words=list(stop_words),
        token_pattern=r'\b\w+\b',
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
        if job_tfidf[i] > 0 and resume_tfidf[i] > 0:
            common_terms.append((term, job_tfidf[i] + resume_tfidf[i]))
    
    common_terms.sort(key=lambda x: x[1], reverse=True)
    matches = [term for term, _ in common_terms][:top_n]
    
    return matches if matches else ["No significant matches"]

def rank_resumes(job_desc_text, keywords, top_n, uploaded_resumes=None):
    """Rank resumes against a job description."""
    model = load_model()
    
    job_desc_text = clean_text(job_desc_text.strip() or "No content")
    job_vector = vectorize_texts_sbert([job_desc_text], model)[0]
    
    resume_texts = []
    resume_filenames = []
    resume_emails = []
    for resume in uploaded_resumes:
        filename = resume.filename
        if filename.endswith(".txt"):
            text = resume.read().decode("utf-8", errors="ignore").strip() or "No content"
        elif filename.endswith(".pdf"):
            text = extract_text_from_pdf(resume)
        else:
            continue
        email = extract_email(text)
        resume_texts.append(text)
        resume_filenames.append(filename)
        resume_emails.append(email)
    
    if not resume_texts:
        return None, "No valid resumes uploaded."
    
    df = pd.DataFrame({
        "ID": resume_filenames,
        "Resume_str": resume_texts,
        "email": resume_emails
    })
    
    # Filter resumes by keywords
    df = filter_resumes(df, keywords, job_desc_text)
    
    # Cap top_n at available resumes
    top_n = min(top_n, len(df))
    
    resume_vectors = vectorize_texts_sbert(df["Resume_str"].tolist(), model)
    similarities = cosine_similarity(resume_vectors, job_vector.reshape(1, -1)).flatten()
    df["similarity"] = similarities
    
    top_resumes = df.sort_values(by="similarity", ascending=False)[["ID", "similarity", "email"]].head(top_n)
    
    top_resumes["key_matches"] = [
        extract_key_matches(df[df["ID"] == row["ID"]]["Resume_str"].iloc[0], job_desc_text)
        for _, row in top_resumes.iterrows()
    ]
    
    return top_resumes, None
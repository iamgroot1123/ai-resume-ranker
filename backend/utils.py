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
from typing import Dict, Any, List, Tuple, Optional
from io import BytesIO
import base64
import openai
import json
import time

nltk.download('stopwords', quiet=True)

# ---------------------------------------------------------------------------
# Model — module-level singleton (no Streamlit cache, pure Python)
# ---------------------------------------------------------------------------
_model: Optional[SentenceTransformer] = None


def load_model_once() -> Optional[SentenceTransformer]:
    """Load the SBERT model once and cache it as a module-level singleton."""
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
            print("[INFO] SBERT model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load SBERT model: {e}")
            _model = None
    return _model


# ---------------------------------------------------------------------------
# Text Extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_like) -> str:
    """Extract text from a PDF file-like object (BytesIO or path)."""
    text = ""
    try:
        with pdfplumber.open(file_like) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text.strip() or "No content"
    except Exception:
        try:
            if hasattr(file_like, "seek"):
                file_like.seek(0)
            reader = PyPDF2.PdfReader(file_like)
            for page in reader.pages:
                text += page.extract_text() or ""
            return text.strip() or "No content"
        except Exception:
            return "No content"


def extract_email(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "No email found"
    match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    return match.group(0) if match else "No email found"


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()
    text = re.sub(
        r'\b(professional summary|summary|references|education|certifications)\b.*?(?=\b\w+\b|$)',
        '', text, flags=re.IGNORECASE
    )
    return text or "No content"


# ---------------------------------------------------------------------------
# Section Extraction  (generic — driven by config, not hard-coded labels)
# ---------------------------------------------------------------------------

ALL_SECTIONS = [
    "SKILLS", "TECHNICAL SKILLS", "EXPERIENCE", "WORK HISTORY",
    "PROFESSIONAL EXPERIENCE", "EDUCATION", "PROJECTS", "CERTIFICATION",
    "ACHIEVEMENTS", "SUMMARY", "CONTACT", "REFERENCES",
]


def extract_section_text(text: str, section_name: str, next_section_names: List[str]) -> str:
    """Pull the text block that belongs to `section_name`."""
    section_name_normalized = section_name.strip()
    pattern_start = (
        r'[\r\n]{1,3}\s*[\-\*\.]*\s*'
        + re.escape(section_name_normalized)
        + r'\s*[:\-\.]{0,5}\s*[\r\n]{1,3}'
    )
    stop_patterns = [
        r'\s*' + re.escape(n.strip()) + r'\s*'
        for n in next_section_names
        if n.strip() != section_name_normalized
    ]
    pattern_stop = '|'.join(stop_patterns)
    regex = re.compile(
        pattern_start + r'(.*?)(?=\s*(' + pattern_stop + r')|$)',
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    text_normalized = re.sub(r'[\r\n]+', '\n', text)
    text_normalized = re.sub(r'[\-\*\•]', '', text_normalized)
    match = regex.search(text_normalized)
    if match:
        content = match.group(1).strip()
        return re.sub(r'[\r\n]{2,}', '\n', content).strip()
    return "Not Found"


def parse_resume_sections(text: str) -> Dict[str, str]:
    """Extract Skills, Experience, and Education from raw resume text."""
    # Education
    education_text = extract_section_text(text, "EDUCATION", ALL_SECTIONS)
    if education_text == "Not Found":
        education_text = extract_section_text(text, "ACADEMIC QUALIFICATIONS", ALL_SECTIONS)
    education_pattern = (
        r'((?:B\.?\s?Tech|M\.?\s?S|Ph\.?\s?D|Bachelor|Master|Degree|Diploma|'
        r'College|University)[\s\w\d,&-]*?(?:\d{4}|\d{2,4}[-\/]\d{2,4}))'
    )
    education_mentions = [
        m.strip()
        for m in re.findall(education_pattern, education_text, re.IGNORECASE)
        if m.strip()
    ]

    # Experience
    experience_text = extract_section_text(text, "PROFESSIONAL EXPERIENCE", ALL_SECTIONS)
    if experience_text == "Not Found":
        experience_text = extract_section_text(text, "WORK EXPERIENCE", ALL_SECTIONS)
    if experience_text == "Not Found":
        experience_text = extract_section_text(text, "EXPERIENCE", ALL_SECTIONS)
    experience_pattern = (
        r'(\w+\s+(engineer|developer|analyst|scientist|manager|consultant|intern)'
        r'[\s\w\d,-]*?(at|on|for|\|)\s*[\w\s\&-]+)'
    )
    experience_mentions = [
        m[0].strip()
        for m in re.findall(experience_pattern, experience_text, re.IGNORECASE)
    ]

    # Skills
    skills_text = extract_section_text(text, "SKILLS", ALL_SECTIONS)
    if skills_text == "Not Found":
        skills_text = extract_section_text(text, "TECHNICAL SKILLS", ALL_SECTIONS)
    sub_header_pattern = (
        r'^\s*[\-\*]*(Languages|Libraries/Frameworks|Others|Core Libraries|'
        r'ML/DL Frameworks|Data Libraries|Cloud & DevOps|Tools|Concepts|'
        r'Programming Languages|Core|Deep Learning|Technical)\s*[:\-\.]{0,2}\s*[\r\n]'
    )
    cleaned_skills = re.sub(sub_header_pattern, ' ', skills_text, flags=re.IGNORECASE | re.MULTILINE).strip()
    universal_skills = re.sub(r'[\r\n]+', ', ', cleaned_skills).strip()

    return {
        "education": " | ".join(education_mentions) or "Not specified",
        "experience": " | ".join(experience_mentions) or "Not specified",
        "skills": universal_skills if universal_skills and universal_skills != "Not Found" else "Not specified",
    }


# ---------------------------------------------------------------------------
# LLM Scoring — Resilient with timeout, retry, and model fallback
# ---------------------------------------------------------------------------

# Ordered fallback chain: primary → fallback 1 → fallback 2
# All are free-tier models on OpenRouter
OPENROUTER_FALLBACK_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-3-27b-it:free",
]

# Status codes that are worth retrying (transient failures)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503}

# Hard limits to avoid token overflows on free-tier models (~1000 tokens each)
MAX_INPUT_CHARS = 4000


def _call_single_model(
    client,
    model_name: str,
    prompt: str,
    is_openrouter: bool,
    max_retries: int = 4,
) -> Tuple[float, str]:
    """
    Attempt one model with exponential backoff on retryable errors.
    Raises on non-retryable errors or after exhausting retries.
    """
    from openai import APIStatusError, APITimeoutError, RateLimitError

    retry_delay = 2  # seconds, doubles each attempt

    for attempt in range(max_retries):
        try:
            create_kwargs = dict(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300,      # Tight output cap — we only need a small JSON
                timeout=10,          # Fail fast; don't hang waiting for overloaded providers
            )
            # response_format is OpenAI-only; OpenRouter providers may reject it
            if not is_openrouter:
                create_kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**create_kwargs)
            result_text = response.choices[0].message.content or ""

            # Strip potential markdown fences from the response
            result_text = result_text.strip()
            for fence in ("```json", "```"):
                if result_text.startswith(fence):
                    result_text = result_text[len(fence):]
            if result_text.endswith("```"):
                result_text = result_text[:-3]

            result_json = json.loads(result_text.strip())
            score = float(result_json.get("fit_score", 0.0))
            justification = result_json.get("justification", "Could not parse justification.")
            return score, justification

        except APITimeoutError:
            print(f"[WARN] Timeout on {model_name} (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay}s...")

        except (RateLimitError, APIStatusError) as e:
            status = getattr(e, "status_code", None)
            if status in RETRYABLE_STATUS_CODES:
                print(f"[WARN] {status} on {model_name} (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay}s...")
            else:
                # Non-retryable (e.g. 401, 402) — propagate immediately
                raise

        if attempt < max_retries - 1:
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)  # cap at 60s

    raise Exception(f"{model_name} failed after {max_retries} attempts.")


def get_llm_score_and_justification(
    resume_text: str, job_desc_text: str, api_key: str, model_name: str = "gpt-3.5-turbo-1106"
) -> Tuple[float, str]:
    """
    Score a resume against a job description using an LLM.

    For OpenRouter keys/models: uses a fallback chain across free-tier models.
    Falls back to the next model if the current one times out, rate-limits,
    or returns a 5xx error.
    """
    if not api_key:
        return 0.0, "API key not provided."

    from openai import OpenAI

    is_openrouter = api_key.startswith("sk-or-") or "/" in model_name

    if is_openrouter:
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://resumeiq.app",
                "X-Title": "ResumeIQ",
            }
        )
        # For OpenRouter: always try the full fallback chain starting from the
        # user's chosen model, then the built-in fallbacks
        model_chain = [model_name] + [
            m for m in OPENROUTER_FALLBACK_MODELS if m != model_name
        ]
    else:
        client = OpenAI(api_key=api_key)
        model_chain = [model_name]   # OpenAI: use exactly what the user picked

    # Truncate inputs to avoid token overflow on free-tier models
    resume_truncated = resume_text[:MAX_INPUT_CHARS]
    jd_truncated = job_desc_text[:MAX_INPUT_CHARS]

    prompt = f"""You are an expert technical recruiter. Evaluate the candidate's resume against the job description.
Provide a "fit_score" from 1.0 to 10.0 (10.0 = perfect match).
Also provide a concise "justification" (2-3 sentences) explaining key strengths or gaps.

Return ONLY a valid JSON object with exactly two keys: "fit_score" and "justification".

--- JOB DESCRIPTION ---
{jd_truncated}

--- RESUME ---
{resume_truncated}"""

    for model in model_chain:
        try:
            print(f"[INFO] Trying model: {model}")
            return _call_single_model(client, model, prompt, is_openrouter)
        except Exception as e:
            status = getattr(e, "status_code", None)
            # Hard stop on auth errors — no point trying fallback models
            if status in (401, 402):
                msg = str(e)
                if "invalid_api_key" in msg or "401" in msg:
                    return 0.0, f"LLM API call failed: {msg}"
                return 0.0, f"LLM API call failed: {msg}"
            print(f"[WARN] Model {model} failed: {e}. Trying next fallback...")

    return 0.0, "All LLM models failed — SBERT fallback will be used."


# ---------------------------------------------------------------------------
# Vectorization & Similarity
# ---------------------------------------------------------------------------

def vectorize_texts_sbert(texts: List[str], model: SentenceTransformer, batch_size: int = 100) -> np.ndarray:
    valid_texts = [
        clean_text(t) if isinstance(t, str) and t.strip() else "No content"
        for t in texts
    ]
    vectors = []
    for i in range(0, len(valid_texts), batch_size):
        batch = valid_texts[i: i + batch_size]
        batch_vectors = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        vectors.append(batch_vectors)
    return np.vstack(vectors)


def filter_resumes(df: pd.DataFrame, keywords: str) -> pd.DataFrame:
    """Hard-filter: only keep documents that contain at least one keyword."""
    if isinstance(keywords, str) and keywords.strip():
        keywords_list = [k.strip().lower() for k in keywords.split(",") if k.strip()]
        if keywords_list:
            mask = df["Resume_str"].str.lower().str.contains(
                "|".join(re.escape(k) for k in keywords_list), na=False
            )
            return df[mask].copy()
    return df


def extract_key_matches(resume_text: str, job_desc_text: str, top_n: int = 3) -> List[str]:
    """Find overlapping key terms between resume and JD using TF-IDF."""
    resume_text = clean_text(resume_text)
    job_desc_text = clean_text(job_desc_text)
    stop_words = set(stopwords.words('english')).union({
        'seeking', 'expertise', 'summary', 'responsibilities',
        'qualifications', 'must', 'have', 'experience', 'ability',
    })
    vectorizer = TfidfVectorizer(
        stop_words=list(stop_words),
        token_pattern=r'\b\w{2,}\b',
        max_features=100,
        ngram_range=(1, 2),
    )
    try:
        tfidf_matrix = vectorizer.fit_transform([job_desc_text, resume_text])
        feature_names = vectorizer.get_feature_names_out()
    except Exception:
        return ["No significant matches"]

    job_tfidf = tfidf_matrix[0].toarray().flatten()
    resume_tfidf = tfidf_matrix[1].toarray().flatten()

    common_terms = [
        (term, job_tfidf[i] + resume_tfidf[i])
        for i, term in enumerate(feature_names)
        if job_tfidf[i] > 0.05 and resume_tfidf[i] > 0.05
    ]
    common_terms.sort(key=lambda x: x[1], reverse=True)
    matches = [t for t, _ in common_terms[:top_n]]
    return [m.replace(" ", "_") for m in matches] if matches else ["No significant matches"]


# ---------------------------------------------------------------------------
# File Processing  (no DB — accepts raw bytes dicts)
# ---------------------------------------------------------------------------

def process_uploaded_files(uploaded_resumes: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Process a list of {"name": str, "bytes": bytes} dicts into a DataFrame.
    No database interaction — pure in-memory processing.
    """
    data = []
    for resume in uploaded_resumes:
        filename: str = resume.get("name", "")
        file_bytes: bytes = resume.get("bytes", b"")

        if not filename or not file_bytes:
            continue

        try:
            if filename.lower().endswith(".txt"):
                text = file_bytes.decode("utf-8", errors="ignore").strip() or "No content"
            elif filename.lower().endswith(".pdf"):
                text = extract_text_from_pdf(BytesIO(file_bytes))
            else:
                continue  # unsupported type

            email = extract_email(text)
            parsed_info = parse_resume_sections(text)

            data.append({
                "ID": filename,
                "Resume_str": text,
                "email": email,
                "parsed_data": parsed_info,
                "file_bytes": file_bytes,
            })
        except Exception as e:
            print(f"[WARN] Error processing '{filename}': {e}")
            continue

    return pd.DataFrame(data) if data else pd.DataFrame()


# ---------------------------------------------------------------------------
# Main Ranking Pipeline  (stateless, no DB)
# ---------------------------------------------------------------------------

def rank_resumes(
    job_desc_text: str,
    keywords: str,
    top_n: int,
    uploaded_resumes: List[Dict[str, Any]],
    model: SentenceTransformer,
    api_key: str,
    use_llm: bool,
    llm_model: str = "gpt-3.5-turbo-1106"
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Rank uploaded documents against a query/job description.

    Returns:
        (results_list, error_string_or_None)

    Each result dict contains structured data + base64-encoded original file bytes
    so the frontend can offer stateless downloads without a database.
    """
    if not model:
        return [], "Semantic model not loaded. Please check server configuration."

    df = process_uploaded_files(uploaded_resumes)
    if df.empty:
        return [], "No valid documents could be processed. Check file formats (.txt / .pdf)."

    cleaned_jd = clean_text(job_desc_text.strip() or "No content")
    job_vector = vectorize_texts_sbert([cleaned_jd], model)[0]

    df_filtered = filter_resumes(df, keywords)
    if df_filtered.empty:
        return [], "No documents matched the specified keywords."

    top_n = min(top_n, len(df_filtered))

    # SBERT similarity for all filtered docs
    resume_vectors = vectorize_texts_sbert(df_filtered["Resume_str"].tolist(), model)
    similarities = cosine_similarity(resume_vectors, job_vector.reshape(1, -1)).flatten()
    df_filtered = df_filtered.copy()
    df_filtered["similarity"] = similarities

    # Select best candidates (by SBERT) to send to LLM if needed
    candidates = df_filtered.sort_values(by="similarity", ascending=False).head(top_n)

    results: List[Dict[str, Any]] = []
    for _, row in candidates.iterrows():
        original = df[df["ID"] == row["ID"]].iloc[0]
        original_text = original["Resume_str"]
        parsed_info = original["parsed_data"]
        file_bytes = original["file_bytes"]

        matches = extract_key_matches(original_text, cleaned_jd)

        if use_llm:
            # Add a small delay between calls to respect free tier RPM limits
            if results: # Don't sleep for the very first one
                time.sleep(2.0)
                
            rating_10, justification = get_llm_score_and_justification(
                original_text, job_desc_text, api_key, llm_model
            )
            
            # Hard stop: if key is unauthorized (401), abort all and alert user
            if rating_10 == 0.0 and "401" in justification and "invalid_api_key" in justification:
                return [], "Invalid API Key provided. Please check your API key and try again."

            # Catch-all graceful fallback: any LLM failure (rate limit, 402 spend cap,
            # parse error, timeouts, etc.) falls back silently to SBERT scoring.
            if rating_10 == 0.0:
                rating_10 = round(1 + row["similarity"] * 9, 1)
                justification = (
                    f"LLM unavailable — SBERT fallback. "
                    f"Key overlaps: {', '.join(matches).replace('_', ' ')}."
                )
        else:
            rating_10 = round(1 + row["similarity"] * 9, 1)
            justification = (
                f"Strong semantic alignment on: {', '.join(matches).replace('_', ' ')}."
            )

        results.append({
            "id": row["ID"],
            "similarity": float(row["similarity"]),
            "email": row["email"],
            "rating_10": rating_10,
            "key_matches": ", ".join(matches).replace("_", " "),
            "justification": justification,
            "skills": parsed_info["skills"],
            "experience": parsed_info["experience"],
            "education": parsed_info["education"],
            # Embed file bytes so frontend can offer stateless downloads
            "file_bytes_b64": base64.b64encode(file_bytes).decode("utf-8"),
            "file_type": "pdf" if row["ID"].lower().endswith(".pdf") else "txt",
        })

    if not results:
        return [], "No candidates found after filtering and ranking."

    results.sort(key=lambda x: x["rating_10"], reverse=True)
    return results, None

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import os
from tqdm import tqdm
import argparse
import pickle

def load_data(resumes_path):
    """Load all resumes and job descriptions."""
    csv_path = resumes_path / "Resume.csv"
    df = pd.read_csv(csv_path)
    
    df["Resume_str"] = df["Resume_str"].fillna("No content").astype(str)
    training_df = df.copy()
    print(f"Training: Using {len(training_df)} resumes across {len(training_df['Category'].unique())} categories.")
    testing_df = df.copy()
    print(f"Testing: Loaded {len(testing_df)} resumes.")
    
    job_desc_path = resumes_path / "job_descriptions"
    job_descs = {}
    for txt_file in job_desc_path.glob("*.txt"):
        category = txt_file.stem
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
            job_descs[category] = text if text else "No content"
    
    return training_df, testing_df, job_descs

def filter_resumes(df, keywords):
    """Filter resumes containing specific keywords."""
    filtered_df = df[df["Resume_str"].str.lower().str.contains("|".join(keywords), na=False)].copy()
    return filtered_df

def vectorize_texts_sbert(texts, model, batch_size=100):
    """Vectorize texts using Sentence-BERT with batch processing."""
    valid_texts = [t if isinstance(t, str) and t.strip() else "No content" for t in texts]
    vectors = []
    for i in tqdm(range(0, len(valid_texts), batch_size), desc="Encoding batches"):
        batch = valid_texts[i:i + batch_size]
        batch_vectors = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        vectors.append(batch_vectors)
    vectors = np.vstack(vectors)
    print(f"Encoded {len(vectors)} texts with shape {vectors.shape}")
    return vectors, model

def rank_resumes(df, job_descs, valid_categories, model, top_n=5, phase="Training", keywords=None, job_desc_text=None, job_desc_file=None):
    """Rank resumes against job descriptions or custom input."""
    if phase == "Testing" and keywords:
        print(f"{phase}: Filtering resumes with keywords: {keywords}")
        df = filter_resumes(df, keywords)
        if df.empty:
            print(f"{phase}: No resumes match keywords. Using all resumes.")
            df = df.copy()
        print(f"{phase}: Filtered to {len(df)} resumes.")
    
    print(f"{phase}: Preparing texts...")
    df.loc[:, "processed_resume"] = df["Resume_str"].astype(str)
    
    if job_desc_text or job_desc_file:
        if job_desc_file:
            with open(job_desc_file, "r", encoding="utf-8") as f:
                job_desc_text = f.read().strip() or "No content"
        category = Path(job_desc_file).stem if job_desc_file else "custom"
        job_desc_processed = {category: job_desc_text}
        valid_categories = [category]
    else:
        job_desc_processed = {cat: text for cat, text in job_descs.items() if cat in valid_categories}
    
    print(f"{phase}: Vectorizing texts with Sentence-BERT...")
    all_texts = df["processed_resume"].tolist() + list(job_desc_processed.values())
    vectors, _ = vectorize_texts_sbert(all_texts, model)
    
    resume_vectors = vectors[:len(df)]
    job_desc_vectors = vectors[len(df):]
    
    results = {}
    for idx, category in enumerate(tqdm(job_desc_processed.keys(), desc=f"{phase} Ranking")):
        print(f"{phase}: Ranking for {category}...")
        try:
            similarities = cosine_similarity(resume_vectors, job_desc_vectors[idx:idx+1]).flatten()
            df.loc[:, "similarity"] = similarities
        except Exception as e:
            print(f"Error ranking {category}: {e}")
            continue
        if phase == "Training" and category != "custom":
            category_resumes = df[df["Category"] == category].copy()
        else:
            category_resumes = df.copy()
        if not category_resumes.empty:
            top_resumes = category_resumes.sort_values(by="similarity", ascending=False)[["ID", "similarity"]].head(top_n)
            results[category] = top_resumes
        else:
            print(f"{phase}: No resumes found for {category}")
    
    return results

def run_resume_ranker(resumes_df, job_desc_text, top_n=5, keywords=None):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-mpnet-base-v2", device="cpu")

    resumes_df = resumes_df.copy()
    resumes_df["Resume_str"] = resumes_df["Resume_str"].fillna("No content").astype(str)

    results = rank_resumes(
        resumes_df,
        job_descs={ "custom": job_desc_text },
        valid_categories=["custom"],
        model=model,
        top_n=top_n,
        phase="Testing",
        keywords=keywords,
        job_desc_text=job_desc_text
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Resume Ranker with SBERT")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top resumes to return per category")
    parser.add_argument("--job-desc-file", type=str, help="Path to custom job description file for testing")
    parser.add_argument("--keywords", type=str, help="Comma-separated keywords for filtering")
    args = parser.parse_args()
    
    # BASE_PATH = Path("/home/madhukiran/Desktop/Elevate Labs/Project/ai-resume-ranker")
    BASE_PATH = Path(os.getenv("BASE_PATH", "/tmp/ai-resume-ranker"))
    RESUMES_PATH = BASE_PATH / "resumes"
    RESULTS_PATH = BASE_PATH / "Results"
    OUTPUT_DIR = BASE_PATH / "scripts/SBERT"
    EMBEDDINGS_PATH = RESULTS_PATH / "embeddings.npy"
    MODEL_PATH = RESULTS_PATH / "sbert_model.pkl"
    
    if not RESUMES_PATH.exists():
        print(f"Error: Folder {RESUMES_PATH} does not exist.")
        return
    if not RESULTS_PATH.exists():
        print(f"Error: Folder {RESULTS_PATH} does not exist. Creating it...")
        RESULTS_PATH.mkdir(parents=True)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load SBERT model
    model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
    
    if EMBEDDINGS_PATH.exists():
        print("Loading pre-computed embeddings...")
        resume_vectors = np.load(EMBEDDINGS_PATH)
        with open(MODEL_PATH, "rb") as f:
            saved_data = pickle.load(f)
        training_df = saved_data["training_df"]
        print(f"Loaded embeddings for {len(resume_vectors)} resumes.")
    else:
        print("Computing embeddings for all resumes...")
        training_df, _, _ = load_data(RESUMES_PATH)
        resume_texts = training_df["Resume_str"].astype(str).tolist()
        resume_vectors, _ = vectorize_texts_sbert(resume_texts, model)
        np.save(EMBEDDINGS_PATH, resume_vectors)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"training_df": training_df}, f)
        print(f"Saved embeddings to {EMBEDDINGS_PATH}")
    
    training_df, testing_df, job_descs = load_data(RESUMES_PATH)
    
    training_categories = training_df["Category"].unique().tolist()
    training_results = rank_resumes(
        training_df, job_descs, training_categories, model, top_n=args.top_n, phase="Training"
    )
    
    if args.job_desc_file:
        keywords = args.keywords.split(",") if args.keywords else None
        testing_results = rank_resumes(
            testing_df, job_descs, [], model, top_n=args.top_n, phase="Testing",
            keywords=keywords, job_desc_file=args.job_desc_file
        )
    else:
        testing_categories = [
            "software-developer", "data-analyst", "data-scientist", "machine-learning-engineer",
            "artificial-intelligence-engineer", "backend-developer", "cloud-engineer", "ai-ml-engineer"
        ]
        testing_keywords = {
            "software-developer": ["software", "programming", "python", "java"],
            "data-analyst": ["data", "analytics", "sql", "excel"],
            "data-scientist": ["data", "machine learning", "statistics", "python"],
            "machine-learning-engineer": ["machine learning", "tensorflow", "pytorch", "python"],
            "artificial-intelligence-engineer": ["ai", "artificial intelligence", "neural networks"],
            "backend-developer": ["backend", "server", "api", "database"],
            "cloud-engineer": ["cloud", "aws", "azure", "docker"],
            "ai-ml-engineer": ["ai", "machine learning", "deep learning"]
        }
        testing_results = {}
        for category in testing_categories:
            print(f"\nTesting for {category}...")
            category_results = rank_resumes(
                testing_df, job_descs, [category], model, top_n=args.top_n,
                phase="Testing", keywords=testing_keywords.get(category, [])
            )
            testing_results.update(category_results)
    
    print("\nTraining Results:")
    for category, top_resumes in training_results.items():
        print(f"Top {args.top_n} resumes for {category}:")
        print(top_resumes.to_string(index=False))
    
    print("\nTesting Results:")
    for category, top_resumes in testing_results.items():
        print(f"Top {args.top_n} resumes for {category}:")
        print(top_resumes.to_string(index=False))
    
    output_path = OUTPUT_DIR / "training_results.csv"
    training_all = pd.concat([df.assign(Category=cat) for cat, df in training_results.items()], ignore_index=True)
    training_all.to_csv(output_path, index=False)
    print(f"\nTraining results saved to {output_path}")
    
    output_path = OUTPUT_DIR / "testing_results.csv"
    testing_all = pd.concat([df.assign(Category=cat) for cat, df in testing_results.items()], ignore_index=True)
    testing_all.to_csv(output_path, index=False)
    print(f"Testing results saved to {output_path}")

if __name__ == "__main__":
    main()
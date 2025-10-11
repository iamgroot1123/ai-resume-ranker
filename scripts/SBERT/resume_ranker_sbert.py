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
    print(f"Loaded {len(df)} resumes across {len(df['Category'].unique())} categories.")
    
    # Load job descriptions
    job_desc_path = resumes_path / "job_descriptions"
    job_descs = {}
    for txt_file in job_desc_path.glob("*.txt"):
        category = txt_file.stem
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
            job_descs[category] = text if text else "No content"
    
    # Use the same DataFrame for both training and testing/evaluation runs
    return df.copy(), df.copy(), job_descs

def filter_resumes(df, keywords):
    """Filter resumes containing specific keywords."""
    if not keywords:
        return df
    keywords_list = [k.strip().lower() for k in keywords]
    filtered_df = df[df["Resume_str"].str.lower().str.contains("|".join(keywords_list), na=False)].copy()
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

def rank_resumes(df, job_vectors, job_desc_categories, resume_vectors, top_n=5, phase="Evaluation", keywords=None):
    """Rank resumes against job description vectors."""
    
    # Pre-filter step for Testing/Evaluation runs
    if phase == "Evaluation" and keywords:
        print(f"{phase}: Filtering resumes with keywords...")
        df_filtered = filter_resumes(df, keywords)
        
        # Determine the subset of resume vectors that correspond to the filtered resumes
        if not df_filtered.empty and len(df_filtered) < len(df):
            # Map filtered IDs back to the original index for slicing the vectors
            original_indices = df[df['ID'].isin(df_filtered['ID'])].index.tolist()
            current_resume_vectors = resume_vectors[original_indices]
            df = df_filtered.copy()
        elif df_filtered.empty:
            print(f"{phase}: No resumes match keywords. Skipping ranking for this category.")
            return {} # Skip this category if no resumes match
        else:
            current_resume_vectors = resume_vectors
    else:
        current_resume_vectors = resume_vectors

    results = {}
    for idx, category in enumerate(tqdm(job_desc_categories, desc=f"{phase} Ranking")):
        print(f"{phase}: Ranking for {category}...")
        try:
            # Calculate similarity for only the current subset of resumes
            similarities = cosine_similarity(current_resume_vectors, job_vectors[idx:idx+1]).flatten()
            
            # Create a temporary DataFrame for this category's results
            temp_df = df.copy()
            temp_df.loc[:, "similarity"] = similarities
            
            # For "Training" (baseline) runs, filter to the correct category
            if phase == "Training":
                category_resumes = temp_df[temp_df["Category"] == category].copy()
            else:
                category_resumes = temp_df.copy() # Use all resumes for testing/custom
                
            if not category_resumes.empty:
                top_resumes = category_resumes.sort_values(by="similarity", ascending=False)[["ID", "similarity"]].head(top_n)
                results[category] = top_resumes
            else:
                print(f"{phase}: No relevant resumes found for {category}")
        except Exception as e:
            print(f"Error ranking {category}: {e}")
            continue
    
    return results

# This function is now mostly redundant since Flask uses utils.py, 
# but we keep it here to maintain compatibility with the original project skeleton's CLI usage.
def run_resume_ranker(resumes_df, job_desc_text, top_n=5, keywords=None):
    # This block should ideally not be used in the CLI script, as the main() function handles
    # the training/testing pipeline. We retain it primarily for conceptual completeness if 
    # the user still wants to use it outside the Flask app.
    
    model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
    resumes_df = resumes_df.copy()
    resumes_df["Resume_str"] = resumes_df["Resume_str"].fillna("No content").astype(str)

    # Encode texts on the fly
    job_vector, _ = vectorize_texts_sbert([job_desc_text], model)
    resume_vectors, _ = vectorize_texts_sbert(resumes_df["Resume_str"].tolist(), model)
    
    job_desc_categories = ["custom"]
    
    # Simplified ranking call using the on-the-fly vectors
    results = rank_resumes(
        resumes_df, 
        job_vectors, 
        job_desc_categories, 
        resume_vectors, 
        top_n=top_n, 
        phase="Evaluation", 
        keywords=keywords.split(",") if keywords else None
    )
    return results

def main():
    parser = argparse.ArgumentParser(description="Resume Ranker with SBERT")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top resumes to return per category")
    parser.add_argument("--skip-training", action="store_true", help="Skip the baseline training evaluation run.")
    args = parser.parse_args()
    
    BASE_PATH = Path(os.getenv("BASE_PATH", Path(__file__).resolve().parent.parent.parent)) # Project Root
    RESUMES_PATH = BASE_PATH / "resumes"
    RESULTS_PATH = BASE_PATH / "Results"
    OUTPUT_DIR = BASE_PATH / "scripts/SBERT"
    EMBEDDINGS_PATH = RESULTS_PATH / "embeddings.npy"
    MODEL_PATH = RESULTS_PATH / "sbert_model.pkl"
    
    if not RESUMES_PATH.exists():
        print(f"Error: Folder {RESUMES_PATH} does not exist.")
        return
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load SBERT model
    model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
    
    training_df, testing_df, job_descs = load_data(RESUMES_PATH)
    all_resumes_df = training_df.copy() # Use a single df for vector generation

    # --- Embedding Logic (Centralized) ---
    if EMBEDDINGS_PATH.exists():
        print("Loading pre-computed embeddings...")
        resume_vectors = np.load(EMBEDDINGS_PATH)
    else:
        print("Computing embeddings for all resumes...")
        resume_texts = all_resumes_df["Resume_str"].astype(str).tolist()
        resume_vectors, _ = vectorize_texts_sbert(resume_texts, model)
        np.save(EMBEDDINGS_PATH, resume_vectors)
        print(f"Saved embeddings to {EMBEDDINGS_PATH}")
    
    # Encode all job descriptions for batch evaluation
    job_desc_categories = list(job_descs.keys())
    job_desc_texts = list(job_descs.values())
    job_vectors, _ = vectorize_texts_sbert(job_desc_texts, model)
    
    # --- Training/Baseline Evaluation ---
    if not args.skip_training:
        print("\n--- Starting Training/Baseline Evaluation ---")
        training_categories = training_df["Category"].unique().tolist()
        
        # Filter JD vectors and categories to only those present in training data
        train_jd_indices = [job_desc_categories.index(c) for c in training_categories if c in job_desc_categories]
        train_job_vectors = job_vectors[train_jd_indices]
        train_job_categories = [job_desc_categories[i] for i in train_jd_indices]

        # Use the full, unfiltered resume_vectors and all_resumes_df for training ranking
        training_results = rank_resumes(
            all_resumes_df, train_job_vectors, train_job_categories, resume_vectors, top_n=args.top_n, phase="Training"
        )
        
        print("\nTraining Results:")
        training_all = pd.DataFrame()
        for category, top_resumes in training_results.items():
            print(f"Top {args.top_n} resumes for {category}:")
            print(top_resumes.to_string(index=False))
            training_all = pd.concat([training_all, top_resumes.assign(Category=category)], ignore_index=True)
            
        output_path = OUTPUT_DIR / "training_results.csv"
        training_all.to_csv(output_path, index=False)
        print(f"\nTraining results saved to {output_path}")

    # --- Testing/Generalization Evaluation ---
    print("\n--- Starting Testing/Generalization Evaluation ---")
    testing_categories = [
        "software-developer", "data-analyst", "data-scientist", "machine-learning-engineer",
        "artificial-intelligence-engineer", "backend-developer", "cloud-engineer", "ai-ml-engineer"
    ]
    testing_keywords = {
        "software-developer": ["software", "programming", "python", "java"],
        "data-analyst": ["data", "analytics", "sql", "excel"],
        "data-scientist": ["data science", "machine learning", "statistics", "python"],
        "machine-learning-engineer": ["machine learning", "tensorflow", "pytorch", "python"],
        "artificial-intelligence-engineer": ["ai", "artificial intelligence", "neural networks", "deep learning"],
        "backend-developer": ["backend", "server", "api", "database", "django", "node"],
        "cloud-engineer": ["cloud", "aws", "azure", "docker", "kubernetes"],
        "ai-ml-engineer": ["ai", "machine learning", "deep learning", "nlp"]
    }
    testing_results = {}
    testing_all = pd.DataFrame()

    for category in testing_categories:
        # Find the JD vector for the current testing category
        try:
            jd_idx = job_desc_categories.index(category)
            job_vector_subset = job_vectors[jd_idx:jd_idx+1]
        except ValueError:
            print(f"Warning: Job description file for {category} not found. Skipping.")
            continue
            
        keywords = testing_keywords.get(category, [])
        print(f"\nTesting for {category} (Keywords: {', '.join(keywords) or 'None'})...")
        
        category_results = rank_resumes(
            all_resumes_df.copy(), # Pass a copy to avoid modification
            job_vector_subset,
            [category], 
            resume_vectors,
            top_n=args.top_n,
            phase="Evaluation", 
            keywords=keywords
        )
        
        testing_results.update(category_results)
        if category in category_results:
            top_resumes = category_results[category]
            print(f"Top {args.top_n} resumes for {category}:")
            print(top_resumes.to_string(index=False))
            testing_all = pd.concat([testing_all, top_resumes.assign(Category=category)], ignore_index=True)

    output_path = OUTPUT_DIR / "testing_results.csv"
    testing_all.to_csv(output_path, index=False)
    print(f"\nTesting results saved to {output_path}")


if __name__ == "__main__":
    main()
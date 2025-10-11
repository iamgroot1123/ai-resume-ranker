import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import os
from tqdm import tqdm
import argparse
import re
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded for NLTK (if not already done by utils.py)
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english')).union({'seeking', 'expertise', 'summary', 'responsibilities', 'qualifications', 'must', 'have', 'experience', 'ability'})

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

def clean_text_for_tfidf(text):
    """Clean text by removing noise, normalizing, and removing stop words."""
    if not isinstance(text, str):
        return "No content"
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()
    # Remove large, non-contextual sections
    text = re.sub(r'\b(professional summary|summary|references|education|certifications)\b.*?(?=\b\w+\b|$)', '', text, flags=re.IGNORECASE)
    
    # Simple tokenization and stop word removal for TF-IDF
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS and len(word) > 1]
    
    return " ".join(tokens) or "No content"

def filter_resumes(df, keywords):
    """Filter resumes containing specific keywords."""
    if not keywords:
        return df
    keywords_list = [k.strip().lower() for k in keywords]
    filtered_df = df[df["Resume_str"].str.lower().str.contains("|".join(keywords_list), na=False)].copy()
    return filtered_df

def vectorize_texts(texts, max_features=5000):
    """Vectorize texts using TF-IDF."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectors = vectorizer.fit_transform(texts)
    return vectors, vectorizer

def rank_resumes(df, job_vectors, job_desc_categories, vectorizer, top_n=5, phase="Evaluation", keywords=None):
    """Rank resumes against job description vectors using the common TF-IDF vectorizer."""
    
    # Preprocess and transform resumes using the *common* fitted vectorizer
    df.loc[:, "processed_resume"] = [clean_text_for_tfidf(str(x)) for x in df["Resume_str"]]
    resume_vectors = vectorizer.transform(df["processed_resume"].tolist())

    # Filter step for Evaluation runs
    if phase == "Evaluation" and keywords:
        df_filtered = filter_resumes(df, keywords)
        
        if not df_filtered.empty and len(df_filtered) < len(df):
            # Map filtered IDs back to the original index for slicing the vectors
            original_indices = df[df['ID'].isin(df_filtered['ID'])].index.tolist()
            current_resume_vectors = resume_vectors[original_indices]
            df = df_filtered.copy()
        elif df_filtered.empty:
            return {}
        else:
            current_resume_vectors = resume_vectors
    else:
        current_resume_vectors = resume_vectors

    results = {}
    for idx, category in enumerate(tqdm(job_desc_categories, desc=f"{phase} Ranking")):
        try:
            # Calculate similarity for only the current subset of resumes
            similarities = cosine_similarity(current_resume_vectors, job_vectors[idx:idx+1]).flatten()
            
            temp_df = df.copy()
            temp_df.loc[:, "similarity"] = similarities
            
            if phase == "Training":
                category_resumes = temp_df[temp_df["Category"] == category].copy()
            else:
                category_resumes = temp_df.copy()
                
            if not category_resumes.empty:
                top_resumes = category_resumes.sort_values(by="similarity", ascending=False)[["ID", "similarity"]].head(top_n)
                results[category] = top_resumes
            else:
                print(f"{phase}: No relevant resumes found for {category}")
        except Exception as e:
            print(f"Error ranking {category}: {e}")
            continue
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Resume Ranker with TF-IDF")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top resumes to return per category")
    parser.add_argument("--skip-training", action="store_true", help="Skip the baseline training evaluation run.")
    args = parser.parse_args()
    
    BASE_PATH = Path(os.getenv("BASE_PATH", Path(__file__).resolve().parent.parent.parent)) # Project Root
    RESUMES_PATH = BASE_PATH / "resumes"
    OUTPUT_DIR = BASE_PATH / "scripts/TF-IDF"
    
    if not RESUMES_PATH.exists():
        print(f"Error: Folder {RESUMES_PATH} does not exist.")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    training_df, testing_df, job_descs = load_data(RESUMES_PATH)
    all_resumes_df = training_df.copy()
    
    # 1. Prepare all text for unified vectorization
    all_texts = (
        all_resumes_df["Resume_str"].tolist() + 
        list(job_descs.values())
    )
    processed_texts = [clean_text_for_tfidf(str(text)) for text in tqdm(all_texts, desc="Preprocessing All Text")]
    
    # 2. Fit vectorizer on *all* processed text
    print("\nFitting TF-IDF Vectorizer...")
    vectors, vectorizer = vectorize_texts(processed_texts)
    
    # 3. Split vectors
    total_resumes = len(all_resumes_df)
    job_vectors = vectors[total_resumes:]
    job_desc_categories = list(job_descs.keys())
    
    # --- Training/Baseline Evaluation ---
    if not args.skip_training:
        print("\n--- Starting Training/Baseline Evaluation ---")
        training_categories = training_df["Category"].unique().tolist()
        
        # Filter JD vectors and categories to only those present in training data
        train_jd_indices = [job_desc_categories.index(c) for c in training_categories if c in job_desc_categories]
        train_job_vectors = job_vectors[train_jd_indices]
        train_job_categories = [job_desc_categories[i] for i in train_jd_indices]
        
        training_results = rank_resumes(
            all_resumes_df.copy(), train_job_vectors, train_job_categories, vectorizer, top_n=args.top_n, phase="Training"
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
        try:
            jd_idx = job_desc_categories.index(category)
            job_vector_subset = job_vectors[jd_idx:jd_idx+1]
        except ValueError:
            print(f"Warning: Job description file for {category} not found. Skipping.")
            continue
            
        keywords = testing_keywords.get(category, [])
        print(f"\nTesting for {category} (Keywords: {', '.join(keywords) or 'None'})...")
        
        category_results = rank_resumes(
            all_resumes_df.copy(), 
            job_vector_subset,
            [category], 
            vectorizer, 
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
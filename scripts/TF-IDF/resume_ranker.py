import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np

def load_data(resumes_path, resumes_per_category=5):
    """Load Resume.csv and job descriptions, sampling for training and testing."""
    csv_path = resumes_path / "Resume.csv"
    df = pd.read_csv(csv_path)
    
    # Sample training data: 5 resumes per category
    training_dfs = []
    for category in df["Category"].unique():
        category_df = df[df["Category"] == category]
        if len(category_df) >= resumes_per_category:
            training_dfs.append(category_df.sample(n=resumes_per_category, random_state=42))
        else:
            print(f"Warning: {category} has only {len(category_df)} resumes, including all.")
            training_dfs.append(category_df)
    
    training_df = pd.concat(training_dfs, ignore_index=True)
    print(f"Training: Sampled {len(training_df)} resumes across {len(training_df['Category'].unique())} categories.")
    
    # Testing data: All ENGINEERING resumes
    testing_df = df[df["Category"] == "ENGINEERING"].copy()
    print(f"Testing: Loaded {len(testing_df)} ENGINEERING resumes.")
    
    # Load job descriptions
    job_desc_path = resumes_path / "job_descriptions"
    job_descs = {}
    for txt_file in job_desc_path.glob("*.txt"):
        category = txt_file.stem
        with open(txt_file, "r", encoding="utf-8") as f:
            job_descs[category] = f.read()
    
    return training_df, testing_df, job_descs

def preprocess_text(text, nlp):
    """Preprocess text using SpaCy (tokenize, lemmatize, remove stop words)."""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def filter_resumes(df, keywords):
    """Filter resumes containing specific keywords."""
    filtered_df = df[df["Resume_str"].str.lower().str.contains("|".join(keywords), na=False)].copy()
    return filtered_df

def vectorize_texts(texts, max_features=5000):
    """Vectorize texts using TF-IDF."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectors = vectorizer.fit_transform(texts)
    return vectors, vectorizer

def rank_resumes(df, job_descs, valid_categories, nlp, phase="Training", keywords=None):
    """Rank resumes against job descriptions."""
    # Filter resumes for testing phase if keywords provided
    if phase == "Testing" and keywords:
        print(f"{phase}: Filtering resumes with keywords: {keywords}")
        df = filter_resumes(df, keywords)
        if df.empty:
            print(f"{phase}: No resumes match keywords. Using all resumes.")
            df = df.copy()  # Revert to original if no matches
        print(f"{phase}: Filtered to {len(df)} resumes.")
    
    # Preprocess resumes
    print(f"{phase}: Preprocessing resumes...")
    df.loc[:, "processed_resume"] = [preprocess_text(str(x), nlp) for x in tqdm(df["Resume_str"], desc="Resumes")]
    
    # Preprocess job descriptions
    print(f"{phase}: Preprocessing job descriptions...")
    job_desc_processed = {cat: preprocess_text(text, nlp) for cat, text in tqdm(job_descs.items(), desc="Job Descriptions") if cat in valid_categories}
    
    # Vectorize all texts
    print(f"{phase}: Vectorizing texts...")
    all_texts = df["processed_resume"].tolist() + list(job_desc_processed.values())
    vectors, vectorizer = vectorize_texts(all_texts)
    
    # Split vectors
    resume_vectors = vectors[:len(df)]
    job_desc_vectors = vectors[len(df):]
    
    # Rank resumes
    results = {}
    for idx, category in enumerate(tqdm(job_desc_processed.keys(), desc=f"{phase} Ranking")):
        print(f"{phase}: Ranking for {category}...")
        similarities = cosine_similarity(resume_vectors, job_desc_vectors[idx:idx+1]).flatten()
        df.loc[:, "similarity"] = similarities
        # Filter resumes (training: match category; testing: all ENGINEERING)
        if phase == "Training":
            category_resumes = df[df["Category"] == category].copy()
        else:
            category_resumes = df.copy()  # All ENGINEERING resumes
        if not category_resumes.empty:
            top_resumes = category_resumes.sort_values(by="similarity", ascending=False)[["ID", "similarity"]].head(5)
            results[category] = top_resumes
        else:
            print(f"{phase}: No resumes found for {category}")
    
    return results

def main():
    # Load SpaCy model
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    
    # Your resumes/ folder path (Ubuntu)
    RESUMES_PATH = Path("/home/madhukiran/Desktop/Elevate Labs/Project/ai-resume-ranker/resumes")
    
    if not RESUMES_PATH.exists():
        print(f"Error: Folder {RESUMES_PATH} does not exist.")
        return
    
    # Load data
    training_df, testing_df, job_descs = load_data(RESUMES_PATH, resumes_per_category=5)
    
    # Training: Rank 5 resumes per category against 24 job descriptions
    training_categories = training_df["Category"].unique().tolist()
    training_results = rank_resumes(training_df, job_descs, training_categories, nlp, phase="Training")
    
    # Testing: Rank ENGINEERING resumes against 8 extra job descriptions with keyword filtering
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
            testing_df, job_descs, [category], nlp, phase="Testing", keywords=testing_keywords.get(category, [])
        )
        testing_results.update(category_results)
    
    # Print results
    print("\nTraining Results:")
    for category, top_resumes in training_results.items():
        print(f"Top 5 resumes for {category}:")
        print(top_resumes.to_string(index=False))
    
    print("\nTesting Results:")
    for category, top_resumes in testing_results.items():
        print(f"Top 5 ENGINEERING resumes for {category}:")
        print(top_resumes.to_string(index=False))
    
    # Create output directory
    output_dir = RESUMES_PATH.parent / "scripts/TF-IDF"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    output_path = output_dir / "training_results.csv"
    training_all = pd.concat([df.assign(Category=cat) for cat, df in training_results.items()], ignore_index=True)
    training_all.to_csv(output_path, index=False)
    print(f"\nTraining results saved to {output_path}")
    
    output_path = output_dir / "testing_results.csv"
    testing_all = pd.concat([df.assign(Category=cat) for cat, df in testing_results.items()], ignore_index=True)
    testing_all.to_csv(output_path, index=False)
    print(f"Testing results saved to {output_path}")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import os
from app.utils import load_model_once, rank_resumes, get_all_parsed_resumes
from io import BytesIO

# --- Configuration ---
st.set_page_config(layout="wide", page_title="AI-Powered Resume Ranker")

# --- Model Loading (Cached) ---
model = load_model_once()
if model is None:
    st.error("Model loading failed. Please check dependencies and memory allocation.")


# --- Page Navigation ---
def set_page(page_name):
    st.session_state['page'] = page_name

if 'page' not in st.session_state:
    st.session_state['page'] = 'ranker'


# ==============================================================================
# 1. RANKER PAGE
# ==============================================================================
def ranker_page():
    # --- Initialize variables (FIX) ---
    final_job_desc = None
    resume_files = None
    error = None # <-- INITIALIZE 'error'
    
    st.title("🤖 AI-Powered Resume Ranker")
    st.markdown("Upload a job description and resumes to rank top candidates using **SBERT Semantic Matching**.")

    # --- Sidebar/Navigation ---
    with st.sidebar: # <--- ENSURE THIS BLOCK IS PRESENT
        st.header("Navigation")
        if st.button("View Stored Database 💾", use_container_width=True):
            set_page('database')
        st.markdown("---")
        
        
    # --- Input Form ---
    with st.container(border=True):
        st.header("1. Job Description (Mandatory)")
        
        col1, col2 = st.columns([2, 1])

        with col1:
            job_desc_text = st.text_area(
                "Paste Job Description Text",
                placeholder="Enter job description...",
                height=200,
                key="jd_text_area"
            )
        
        with col2:
            job_desc_file = st.file_uploader(
                "Upload Job Description (.txt only)",
                type=['txt'],
                key="jd_file_uploader"
            )

        if job_desc_file:
            # Read file contents
            job_desc_bytes = job_desc_file.read()
            job_desc_from_file = job_desc_bytes.decode("utf-8", errors="ignore").strip()
            final_job_desc = job_desc_from_file
        elif job_desc_text:
            final_job_desc = job_desc_text
        else:
            final_job_desc = None
            st.warning("Please provide a job description either by typing text or uploading a .txt file.")


    with st.container(border=True):
        st.header("2. Candidates and Ranking Options")

        col3, col4, col5 = st.columns(3)
        
        with col3:
            resume_files = st.file_uploader(
                "Upload Resumes (.txt, .pdf)",
                type=['txt', 'pdf'],
                accept_multiple_files=True
                , key="resume_file_uploader" 
            )
        
        with col4:
            keywords = st.text_input(
                "Keywords (optional)",
                placeholder="e.g., Python, SQL, AWS"
            )
        
        with col5:
            top_n = st.number_input(
                "Number of Top Resumes",
                min_value=1,
                value=5,
                step=1
            )
    
    st.markdown("---")
    
    if st.button("Rank Resumes", type="primary", use_container_width=True):
        
        # --- FIX 1: Initializing rank_error (from previous step) ---
        rank_error = None
        
        # --- FIX 2: Simplification of validation logic for Streamlit file objects ---
        # The Streamlit st.file_uploader list can contain non-None objects even if no file data is read.
        # We rely on the size check, but must ensure the list exists.
        
        # Ensure resume_files is an iterable list
        uploaded_resumes_list = resume_files if resume_files else []

        # Filter: Only keep files that are not None and have a size greater than 0
        valid_resume_files = [f for f in uploaded_resumes_list if f is not None and f.size > 0]


        if not final_job_desc:
            rank_error = "Job Description is mandatory."
        elif not valid_resume_files: # <-- Check the filtered list
            rank_error = "Please upload at least one resume."
        elif model is None:
            rank_error = "The SBERT model failed to load. Cannot rank."
        
        
        if rank_error:
            st.error(rank_error)
        else:
            with st.spinner('Ranking candidates and extracting structured data...'):
                top_resumes_df, error = rank_resumes(
                    job_desc_text=final_job_desc,
                    keywords=keywords,
                    top_n=top_n,
                    uploaded_resumes=valid_resume_files,
                    model=model
                )
            
            # The 'error' returned from rank_resumes is now used here
            if error:
                st.error(error)
            elif top_resumes_df.empty:
                st.warning("No matching resumes found after keyword filtering or ranking.")
            else:
                st.success(f"Successfully ranked {len(top_resumes_df)} resumes.")
                display_results(top_resumes_df, final_job_desc)



def display_results(df, job_desc):
    st.subheader(f"🏆 Top {len(df)} Candidates")
    st.markdown(f"**Job Description Snippet:** _{job_desc[:200]}..._")

    for index, row in df.iterrows():
        rank = index + 1
        
        with st.expander(f"#{rank} | {row['ID']} | Fit Score: {row['rating_10']} / 10", expanded=rank <= 3):
            st.markdown(f"**Email:** `{row['email']}` | **SBERT Similarity:** `{row['similarity']:.4f}`")
            st.markdown(f"**Justification:** {row['justification']}")
            
            # Key Matches
            matches_html = " ".join([f'<span style="background-color: #d1e7dd; color: #0f5132; padding: 4px 8px; border-radius: 4px; margin-right: 5px; font-size: 0.9em; font-weight: 600;">{match.replace("_", " ")}</span>' 
                                     for match in row['key_matches'].split(', ')])
            st.markdown(f"**Key Matching Terms:** {matches_html}", unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Extracted Structured Data (Feature 1)")
            
            col_s, col_e, col_ed = st.columns(3)
            
            # Structured Data (Handling the current "Not specified" gracefully)
            col_s.metric("Skills", "Extracted" if row['skills'] != "Not specified" else "❌ Failed", help=row['skills'])
            col_s.caption(f"Details: {row['skills']}")
            
            col_e.metric("Experience", "Extracted" if row['experience'] != "Not specified" else "❌ Failed", help=row['experience'])
            col_e.caption(f"Details: {row['experience']}")

            col_ed.metric("Education", "Extracted" if row['education'] != "Not specified" else "❌ Failed", help=row['education'])
            col_ed.caption(f"Details: {row['education']}")

    # --- Download ---
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Full Ranking CSV",
        data=csv_buffer.getvalue(),
        file_name="resume_ranking_results.csv",
        mime="text/csv",
        use_container_width=True
    )


# ==============================================================================
# 2. DATABASE VIEWER PAGE
# ==============================================================================
def database_viewer_page():
    st.title("💾 Stored Resumes Database")
    st.markdown("Review all resumes ever parsed and stored, along with the extracted structured data.")

    # --- Sidebar/Navigation ---
    with st.sidebar:
        st.header("Navigation")
        if st.button("Back to Ranker 🤖", use_container_width=True):
            set_page('ranker')
        st.markdown("---")

    # --- Data Retrieval ---
    resumes_data = get_all_parsed_resumes()
    
    if resumes_data:
        st.success(f"Total Resumes Stored: {len(resumes_data)}")
        df_db = pd.DataFrame(resumes_data)
        
        # Format the DataFrame for better presentation
        df_db = df_db.rename(columns={
            'filename': 'File Name',
            'email': 'Email ID',
            'skills': 'Extracted Skills',
            'experience': 'Extracted Experience',
            'education': 'Extracted Education',
            'upload_date': 'Upload Date'
        })
        
        st.dataframe(df_db, use_container_width=True, hide_index=True)
        
        # --- Download Database CSV ---
        csv_buffer = BytesIO()
        df_db.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Full Database CSV",
            data=csv_buffer.getvalue(),
            file_name="parsed_resumes_database.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:
        st.warning("No resumes have been parsed and stored yet. Please use the Ranker page first.")


# ==============================================================================
# MAIN APP EXECUTION
# ==============================================================================
if st.session_state['page'] == 'ranker':
    ranker_page()
elif st.session_state['page'] == 'database':
    database_viewer_page()
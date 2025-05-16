import os
import git
import time
from pathlib import Path

def setup_git_repo(repo_path):
    """Initialize or verify Git repository."""
    repo_path = Path(repo_path).resolve()
    try:
        repo = git.Repo(repo_path)
        print(f"Existing Git repository found at {repo_path}")
    except git.exc.InvalidGitRepositoryError:
        repo = git.Repo.init(repo_path)
        print(f"Initialized new Git repository at {repo_path}")
    return repo

def commit_and_push(repo, commit_message="Auto-commit: Updated project files"):
    """Commit all changes and push to remote."""
    try:
        # Stage all changes
        repo.git.add(A=True)
        # Check if there are staged changes
        if repo.is_dirty(untracked_files=True):
            # Commit changes
            repo.index.commit(commit_message)
            print(f"Committed changes: {commit_message}")
            # Push to remote
            origin = repo.remote(name="origin")
            origin.push()
            print("Pushed changes to GitHub")
        else:
            print("No changes to commit")
    except Exception as e:
        print(f"Error during commit/push: {e}")

def monitor_folder(repo_path, interval=60):
    """Monitor folder for changes and auto-commit/push every 'interval' seconds."""
    repo = setup_git_repo(repo_path)
    print(f"Monitoring folder: {repo_path}")
    print(f"Checking for changes every {interval} seconds...")
    
    while True:
        try:
            commit_and_push(repo)
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped monitoring. Final commit and push...")
            commit_and_push(repo, commit_message="Final auto-commit before stopping")
            break

if __name__ == "__main__":
    # Replace with your local project folder path
    PROJECT_PATH = "C:/Users/2306m/Desktop/Elevate Labs/Project/ai-resume-ranker"  # e.g., "/home/user/ai-resume-ranker"
    
    # Verify project path exists
    if not os.path.exists(PROJECT_PATH):
        print(f"Error: Folder {PROJECT_PATH} does not exist. Please update PROJECT_PATH.")
    else:
        # Start monitoring (checks every 60 seconds)
        monitor_folder(PROJECT_PATH, interval=60)
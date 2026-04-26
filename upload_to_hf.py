from huggingface_hub import HfApi
import os

token = os.getenv("HF_TOKEN", "your_hugging_face_token_here")
repo_id = "shahkavya2307/email-triage-env-meta"

print(f"Connecting to Hugging Face to upload {repo_id}...")

try:
    api = HfApi(token=token)
    
    # Upload everything in this folder, ignoring hidden/large unneeded files
    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=["**/.git/**", "**/.venv/**", ".venv", ".venv/*", "**/.vscode/**", "**/__pycache__/**", "**/*.pyc", ".env", "upload_to_hf.py"],
        commit_message="Upload via Python API to bypass Git network errors"
    )
    print("✅ Upload Complete! Check your Hugging Face Space: https://huggingface.co/spaces/" + repo_id)
except Exception as e:
    print("❌ Upload failed:", e)

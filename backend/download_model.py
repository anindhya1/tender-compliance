from huggingface_hub import snapshot_download

# This downloads the model to a folder named 'local_embedding_model' in your current directory
snapshot_download(
    repo_id="nomic-ai/nomic-embed-text-v1.5",
    local_dir="./local_embedding_model",
    local_dir_use_symlinks=False  # Important: ensures actual files are downloaded, not just links
)

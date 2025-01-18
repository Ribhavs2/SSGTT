from huggingface_hub import login
import os
HF_TOKEN = os.getenv("HF_TOKEN")

from huggingface_hub import snapshot_download

local_model_path = "./models/meta-llama-3"
snapshot_download(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", local_dir=local_model_path)


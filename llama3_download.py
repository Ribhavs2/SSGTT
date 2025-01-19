from dotenv import load_dotenv
import os
from huggingface_hub import snapshot_download

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

from huggingface_hub import login
login(token=HF_TOKEN)

local_model_path = "./models/Llama-3.1-8B-Instruct"
snapshot_download(repo_id="meta-llama/Llama-3.1-8B-Instruct", local_dir=local_model_path)

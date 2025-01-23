# # setup_env.sh
# #!/bin/bash

# # Install Miniforge
# wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
# bash Miniforge3-Linux-aarch64.sh

# # Create and activate the conda environment
# conda create -n llama_env python=3.10 -y
# conda activate llama_env

# # Install required Python packages
# pip install python-dotenv
# pip install transformers huggingface_hub bitsandbytes accelerate

# # Load the specific module (for cluster systems)
# module load python/miniforge3_pytorch/2.5.0

# # Verify PyTorch and CUDA installation
# python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

# setup_env.sh
#!/bin/bash

# Install Miniforge for macOS ARM64
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh -b -p $HOME/miniforge3

# Initialize Conda
source $HOME/miniforge3/etc/profile.d/conda.sh
conda init

# Create and activate the Conda environment
conda create -n llama_env python=3.10 -y
conda activate llama_env

# Install required Python packages
pip install python-dotenv
pip install transformers huggingface_hub bitsandbytes accelerate

# Verify PyTorch and CUDA installation
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

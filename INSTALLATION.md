# Installation Instructions

## Prerequisites
1. Install `docker`
2. Install `anaconda`
3. Existing 12.7 cuda

## Setup Steps
1. Run `jupyter notebook`
2. Install required Python packages:
   ```bash
   pip install dataset tqdm
   ```
3. Install Ollama:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
   Then pull the required models:
   ```bash
   ollama pull llama2
   ollama pull tinyllama
   ```
4. Install Weights & Biases for experiment logging:
   ```bash
   pip install wandb
   wandb login
   ``` 
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the variables
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
wandb_api_key = os.getenv('WANDB_API_KEY')
import os, dotenv

# TODO: Better way to save api keys is to use .env file
# TODO: NEVER COMMIT .ENV FILES TO GIT
# TODO: THIS IS HOW YOU LOOSE MILLIONS OF DOLLARS IN A DAY

# Load environment variables from .env file
dotenv.load_dotenv()

# Access the variables
GEMINI_API_KEY  = os.getenv('GEMINI_API_KEY')
# HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')
# WANDB_API_KEY   = os.getenv('WANDB_API_KEY')

# OR
# GEMINI_API_KEY  = ""
# HF_ACCESS_TOKEN = ""
# WANDB_API_KEY   = ""

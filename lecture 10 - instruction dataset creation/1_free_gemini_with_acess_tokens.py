from google import genai
import os, dotenv

# SAVED API KEY ON TOP TO KEEP TRACK OF IT
# GEMINI_API_KEY = "AIzaSyCEsL5NELX_0ZyG5AHnJCZgkpczCtXLG8Q"
dotenv.load_dotenv()
GEMINI_API_KEY  = os.getenv('GEMINI_API_KEY')

client         = genai.Client(api_key=GEMINI_API_KEY)
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="How does AI work?"
)
print(response.text)
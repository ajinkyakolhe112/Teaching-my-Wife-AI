from google import genai

# SAVED API KEY ON TOP TO KEEP TRACK OF IT
GEMINI_API_KEY = "AIzaSyCEsL5NELX_0ZyG5AHnJCZgkpczCtXLG8Q"


client         = genai.Client(api_key=GEMINI_API_KEY)
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="How does AI work?"
)
print(response.text)
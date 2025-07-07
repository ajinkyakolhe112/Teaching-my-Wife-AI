import openai, base64, os, dotenv
from pathlib import Path

dotenv.load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

client = openai.OpenAI()

text_question     = "What are the foundational models in Deep Learning"
text_completion   = client.chat.completions.create     ( model="gpt-4o-mini", messages =[ {"role": "user", "content": text_question}])

image_description = "a diagram of a neural network with 3 layers"
image_generation  = client.images.generate             ( model="dall-e-3",    prompt= image_description, size="1024x1024", quality="standard",n=1,)

audio_question    = "Is a golden retriever a good family dog?"
audio_response    = client.chat.completions.create     ( model="gpt-4o-audio-preview", messages=[{"role": "user", "content": text_to_audio}], modalities=["text", "audio"], audio={"voice": "alloy", "format": "wav"}, )

text_to_audio     = "Today is a wonderful day to build something people love!"
text_to_speech    = client.audio.speech.create         ( model="tts-1",       input= text_to_audio, voice="alloy", )

transcription     = client.audio.transcriptions.create ( model="whisper-1",   file=text_to_audio)


# Text to Speech related lines
speech_file_path = Path("./").parent / "model response - speech.mp3"
text_to_speech.stream_to_file(speech_file_path)
audio_file       = open(speech_file_path, "rb")
transcription    = client.audio.transcriptions.create( model="whisper-1",   file=audio_file)
print(transcription.text)

# Writing Audio Response to wave file
wav_bytes = base64.b64decode(audio_response.choices[0].message.audio.data)
f = open("audio answer to a question.wav", "wb")
f.write(wav_bytes)
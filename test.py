import os
from google.cloud import vision
from google.cloud import texttospeech
from google.oauth2 import service_account
import torchaudio

torchaudio.set_audio_backend('soundfile')
# Load the service account key
credentials = service_account.Credentials.from_service_account_file('G:/ML PROJ/EchoRx/.gitignore/spatial-lock-389419-d2e8d7a3196b.json')

# Initialize the Vision and Text-to-Speech clients
vision_client = vision.ImageAnnotatorClient(credentials=credentials)
tts_client = texttospeech.TextToSpeechClient(credentials=credentials)

# Path to the image file
image_file_path = 'C:/Users/adway/Downloads/test.png'

# Perform OCR using Google Cloud Vision API
with open(image_file_path, 'rb') as image_file:
    content = image_file.read()

image = vision.Image(content=content)
response = vision_client.text_detection(image=image)
texts = response.text_annotations
ocr_text = texts[0].description if texts else ''

print("OCR Text:")
print(ocr_text)

# Convert the OCR text to speech using Google Cloud Text-to-Speech API
input_text = texttospeech.SynthesisInput(text=ocr_text)
voice = texttospeech.VoiceSelectionParams(language_code='en-US', ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

response = tts_client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
output_file = 'static/output.mp3'

with open(output_file, 'wb') as audio_file:
    audio_file.write(response.audio_content)

print("Audio file saved as:", output_file)

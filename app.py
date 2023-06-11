import os
import numpy
import cv2
from PIL import Image
import torchaudio
import torch
from google.cloud import vision
from google.cloud import texttospeech
from google.oauth2 import service_account
from flask import Flask, render_template, request
import tempfile

tempfile.tempdir = 'temp'
temp_dir = 'temp'
output_file = 'static/output.wav'
torchaudio.set_audio_backend('soundfile')

app = Flask(__name__)

# Configure Google Cloud credentials
credentials = service_account.Credentials.from_service_account_file('G:/ML PROJ/EchoRx/.gitignore/spatial-lock-389419-d2e8d7a3196b.json')

# Create a Google Cloud Vision client
vision_client = vision.ImageAnnotatorClient(credentials=credentials)

# Create a Google Cloud Text-to-Speech client
tts_client = texttospeech.TextToSpeechClient(credentials=credentials)

# Load the pre-trained TTS model and configure the audio settings
audio_resampler = torchaudio.transforms.Resample(48_000, 16_000)


# Function to convert prescription image to spoken speech
def convert_prescription_to_speech(image):
    # Perform OCR using Google Cloud Vision API
    # temp_dir = tempfile.mkdtemp(dir='.', prefix='temp_', suffix='')

    # Save the image to the temporary directory
    temp_image_path = os.path.join(temp_dir, 'temp_image.png')
    image.save(temp_image_path)

    with open(temp_image_path, "rb") as image_file:
        image_content = image_file.read()

    vision_image = vision.Image(content=image_content)
    response = vision_client.text_detection(image=vision_image)
    texts = response.text_annotations
    ocr_text = texts[0].description if texts else ''

    # Synthesize speech using Google Cloud Text-to-Speech API
    synthesis_input = texttospeech.SynthesisInput(text=ocr_text)

    voice = texttospeech.VoiceSelectionParams(language_code='en-IN', ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)

    response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    audio_content = response.audio_content

    # Save the synthesized speech to a WAV file
    with open(output_file, 'wb') as audio_file:
        audio_file.write(audio_content)


# Function to capture an image using OpenCV
def capture_image():
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    if ret:
        # Convert BGR image to grayscale
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = Image.fromarray(image)
        return image

    return None


# Routes (same as before)
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    # Check if an image was uploaded
    if 'image' not in request.files:
        return render_template('index.html', error='No image file provided.')

    image = request.files['image']

    # Check if the file is an image
    if image.filename == '':
        return render_template('index.html', error='No image file provided.')

    # Check the file format
    allowed_extensions = ['.jpg', '.jpeg', '.png']
    if not any(image.filename.lower().endswith(ext) for ext in allowed_extensions):
        return render_template('index.html', error='Invalid image file format.')

    try:
        # Convert the uploaded image to PIL format
        pil_image = Image.open(image).convert('L')

        # Perform OCR on the uploaded image using Google Cloud Vision API and convert to speech
        convert_prescription_to_speech(pil_image)

        return render_template('result.html', audio_file='static/output.wav')

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return render_template('index.html', error=error_message)


if __name__ == '__main__':
    app.run(debug=True)

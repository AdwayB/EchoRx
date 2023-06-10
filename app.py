import os
import cv2
from PIL import Image
import torchaudio
import torch
from google.cloud import vision
from google.oauth2 import service_account
from gtts import gTTS
from flask import Flask, render_template, request
import tempfile

tempfile.tempdir = 'temp'

app = Flask(__name__)


credentials = service_account.Credentials.from_service_account_file('G:/ML PROJ/EchoRx/.gitignore/spatial-lock-389419-4d2ce3eff0a9.json')
vision_client = vision.ImageAnnotatorClient(credentials=credentials)

# Load the pre-trained TTS model and configure the audio settings
audio_config = torchaudio.transforms.Resample(48_000, 16_000)


# Function to convert prescription image to spoken speech
def convert_prescription_to_speech(image):
    # Perform OCR using Google Cloud Vision API
    # temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')

    # Create a temporary directory with the necessary permissions
    temp_dir = tempfile.mkdtemp(dir='.', prefix='temp_', suffix='')

    # Save the image to the temporary directory
    temp_image_path = os.path.join(temp_dir, 'temp_image.png')
    image.save(temp_image_path)

    with tempfile.NamedTemporaryFile(suffix=".png", dir=temp_image_path) as temp_image:
        image.save(temp_image.name)
        with open(temp_image.name, "rb") as image_file:
            image_content = image_file.read()

        vision_image = vision.Image(content=dict(image_content))
        response = vision_client.text_detection(image=vision_image)
        texts = response.text_annotations
        ocr_text = texts[0].description if texts else ''

    # Convert the OCR text to speech using gTTS
    tts = gTTS(text=ocr_text, lang='en')
    os.remove(temp_image_path)
    os.rmdir(temp_dir)
    # Save the synthesized speech to a WAV file
    output_file = 'output.wav'
    tts.save(output_file)

    # Convert audio to numpy array and resample to 16kHz
    audio_resampled, _ = torchaudio.load(output_file, num_frames=-1, normalize=True)
    audio_resampled = audio_config(torch.from_numpy(audio_resampled.numpy())).numpy()

    return audio_resampled


# Function to capture an image using OpenCV
def capture_image():
    camera = cv2.VideoCapture(0)  # Use the default camera or specify the device ID
    ret, frame = camera.read()
    if ret:
        # Convert BGR image to grayscale
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Convert grayscale image to PIL Image format
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

    try:
        # Convert the uploaded image to PIL format
        pil_image = Image.open(image).convert('L')

        # Perform OCR on the uploaded image using Google Cloud Vision API and convert to speech
        audio_output = convert_prescription_to_speech(pil_image)

        # Save the synthesized speech to a WAV file
        output_file = 'output.wav'
        torchaudio.save(output_file, torch.from_numpy(audio_output), 16000)

        return render_template('result.html', audio_file=output_file)

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return render_template('index.html', error=error_message)


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request
import cv2
from PIL import Image
import pytesseract
import tensorflow as tf
import numpy as np
import torch
import torchaudio
from transformers import T5Tokenizer, TFT5ForConditionalGeneration

app = Flask(__name__)

# Load OCR model (Tesseract) and TTS model (T5-based)
pytesseract.pytesseract.tesseract_cmd = 'path/to/tesseract'
tokenizer = T5Tokenizer.from_pretrained('t5-base')
tts_model = TFT5ForConditionalGeneration.from_pretrained('t5-base')


# Load the pre-trained TTS model and configure the audio settings
tts_model._make_predict_function()  # For TensorFlow 1.x compatibility
audio_config = torchaudio.transforms.Resample(48_000, 16_000)


# Function to convert prescription image to spoken speech
def convert_prescription_to_speech(image):
    # Perform OCR to extract text from the prescription image
    prescription_text = pytesseract.image_to_string(image)

    # Preprocess the text and convert it to speech
    input_ids = tokenizer.encode(prescription_text, return_tensors='tf')
    input_ids = tf.convert_to_tensor(input_ids)

    # Generate speech from the input text
    audio = tts_model.generate(input_ids)
    audio = tf.squeeze(audio, axis=0)

    # Convert audio to numpy array and resample to 16kHz
    audio_np = np.array(audio)
    audio_resampled = audio_config(torch.from_numpy(audio_np)).numpy()

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


# Routes
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
    
    if not allowed_file(image.filename):
        return render_template('index.html', error='Invalid file format. Only images are allowed.')
    
    try:
        # Process the uploaded image
        image = Image.open(image).convert('L')  # Convert to grayscale
        image = image.resize((224, 224))  # Resize to match TTS model input size
        
        # Convert PIL image to TensorFlow tensor
        image_tensor = tf.convert_to_tensor(np.array(image), dtype=tf.float32)
        image_tensor /= 255.  # Normalize pixel values to [0, 1]

        # Convert single image tensor to a batch of size 1
        image_tensor = tf.expand_dims(image_tensor, axis=0)

        # Generate speech from the prescription image
        audio_output = convert_prescription_to_speech(image_tensor)

        # Save the synthesized speech to a WAV file
        output_file = 'static/output.wav'
        torchaudio.save(output_file, torch.from_numpy(audio_output), 16000)

        return render_template('result.html', audio_file=output_file)

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return render_template('index.html', error=error_message)


# Helper function to check allowed file extensions
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(debug=True)

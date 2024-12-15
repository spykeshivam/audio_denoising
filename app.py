from flask import Flask, request, render_template, send_file
import os
from werkzeug.utils import secure_filename
from models.denoising_model import load_model, denoise_audio
from utils.audio_processing import load_audio, save_audio

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['OUTPUT_FOLDER'] = 'outputs/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load the model globally
model = load_model("models/pretrained_model.pth")

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio_file' not in request.files:
        return "No file part", 400

    file = request.files['audio_file']
    if file.filename == '':
        return "No selected file", 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    # Process the audio
    audio, sr = load_audio(input_path)
    denoised_audio = denoise_audio(model, audio, sr)

    # Save the output in the folder
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"denoised_{filename}")
    save_audio(denoised_audio, sr, output_path)

    return send_file(output_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)

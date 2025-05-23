from flask import Flask, request, send_file, jsonify
from TTS.api import TTS
import whisper
import uuid
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize models
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
stt = whisper.load_model("base")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/api/tts", methods=["POST"])
def text_to_speech():
    text = request.json.get("text")
    if not text:
        return jsonify({"error": "Missing text"}), 400
    
    filename = f"{secure_filename(str(uuid.uuid4()))}.wav"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        tts.tts_to_file(text=text, file_path=filepath)
        return send_file(filepath, as_attachment=True, mimetype='audio/wav')
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route("/api/stt", methods=["POST"])
def speech_to_text():
    if 'file' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400
    
    filename = f"{secure_filename(str(uuid.uuid4()))}.wav"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        result = stt.transcribe(filepath)
        return jsonify({"text": result["text"]})
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/')
def health_check():
    return jsonify({"status": "healthy", "service": "TTS/STT API"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Important for Render
    app.run(host="0.0.0.0", port=port)  # Render requires 0.0.0.0
from flask import Flask, request, send_file, jsonify
from TTS.api import TTS
import whisper
import uuid
import os

app = Flask(__name__)

# Initialize models
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
stt = whisper.load_model("base")

@app.route("/api/tts", methods=["POST"])
def text_to_speech():
    text = request.json.get("text")
    if not text:
        return jsonify({"error": "Missing text"}), 400
    filename = f"{uuid.uuid4()}.wav"
    tts.tts_to_file(text=text, file_path=filename)
    return send_file(filename, as_attachment=True)

@app.route("/api/stt", methods=["POST"])
def speech_to_text():
    if 'file' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    file = request.files['file']
    filepath = f"{uuid.uuid4()}.wav"
    file.save(filepath)
    result = stt.transcribe(filepath)
    os.remove(filepath)
    return jsonify({"text": result["text"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

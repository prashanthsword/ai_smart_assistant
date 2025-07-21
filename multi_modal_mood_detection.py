import sounddevice as sd
import numpy as np
import torch
import cv2
from fer import FER
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import warnings
warnings.filterwarnings("ignore")

# === CONFIG ===
your_name = "Prashanth"
sample_rate = 16000
duration = 5  # seconds

# === LOAD MODELS ===
print("🔧 Loading models...")
face_detector = FER(mtcnn=True)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
print("✅ Models loaded.")

# === FACIAL EMOTION DETECTION ===
def detect_facial_emotion():
    print("📸 Capturing face...")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return "unknown"

    emotion_data = face_detector.detect_emotions(frame)
    if emotion_data:
        top_emotion = emotion_data[0]["emotions"]
        mood = max(top_emotion, key=top_emotion.get)
        confidence = round(top_emotion[mood] * 100, 2)
        print(f"🙂 Facial Emotion: {mood} ({confidence}%)")
        return mood
    else:
        print("😐 No face detected.")
        return "unknown"

# === VOICE EMOTION DETECTION ===
def detect_voice_emotion():
    print("🎙️ Speak now (5 seconds)...")
    try:
        audio = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        if not isinstance(audio, np.ndarray) or audio.shape[0] == 0:
            raise ValueError("Empty audio")

        inputs = processor(audio, return_tensors="pt", sampling_rate=sample_rate)
        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
        print(f"🗣️ Transcribed: {transcription.lower()}")

        # Simple emotion classification based on keywords
        text = transcription.lower()
        if any(word in text for word in ["happy", "excited", "good", "great"]):
            mood = "happy"
        elif any(word in text for word in ["sad", "tired", "bad", "depressed"]):
            mood = "sad"
        elif any(word in text for word in ["angry", "frustrated", "irritated"]):
            mood = "angry"
        else:
            mood = "neutral"
        print(f"🧠 Voice-based Mood: {mood}")
        return mood
    except Exception as e:
        print(f"⚠️ Voice emotion error: {e}")
        return "unknown"

# === FINAL DECISION ===
def decide_final_mood(face, voice):
    if face != "unknown":
        return face
    elif voice != "unknown":
        return voice
    else:
        return "neutral"

# === MAIN ===
if __name__ == "__main__":
    print("🤖 Starting Multi-Modal Mood Detector...")
    print(f"👋 Hello {your_name}, I’m ready to assist you today!")

    fe = detect_facial_emotion()
    ve = detect_voice_emotion()

    final = decide_final_mood(fe, ve)
    print(f"\n🎯 Final Detected Mood: {final.upper()}")

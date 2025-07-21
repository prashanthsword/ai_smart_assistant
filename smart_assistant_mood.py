import cv2
import torch
import pyttsx3
import speech_recognition as sr
from fer import FER
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import sounddevice as sd
import numpy as np
import subprocess
import openai  # Only if using ChatGPT

# --------- CONFIG ---------
USE_CHATGPT = False  # Set to True to use ChatGPT (requires internet + API key)
openai.api_key = "sk-..."  # Add your OpenAI key if using ChatGPT

# --------- INIT ---------
engine = pyttsx3.init()
emotion_detector = FER(mtcnn=True)
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

def speak(text):
    engine.say(text)
    engine.runAndWait()

def get_mood_from_face():
    cap = cv2.VideoCapture(0)
    print("📸 Capturing face...")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "neutral"
    
    emotions = emotion_detector.detect_emotions(frame)
    if emotions:
        top_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
        print(f"🙂 Facial Emotion: {top_emotion}")
        return top_emotion
    return "neutral"

def record_audio(duration=5):
    print("🎙️ Speak now...")
    recording = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='float32')
    sd.wait()
    return recording.flatten()

def transcribe_audio(audio):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0].lower()

def ask_chatgpt(text, mood):
    print("🤖 ChatGPT Thinking...")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"You are an assistant who adapts to the user's mood: {mood}."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content.strip()

def ask_llama(text, mood):
    prompt = f"[Mood: {mood}]\nUser: {text}\nAI:"
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    response = result.stdout.decode().strip()
    return response.split(">>>")[-1].strip()

# --------- MAIN LOOP ---------
print("🤖 Smart Assistant with Mood Detection is ready. Say 'exit' to stop.")

while True:
    mood = get_mood_from_face()
    audio = record_audio()
    try:
        text = transcribe_audio(audio)
    except Exception as e:
        print(f"❌ Voice Recognition Error: {e}")
        continue

    if text in ["exit", "quit", "bye"]:
        speak("Goodbye!")
        break

    print(f"🗣️ You said: {text}")
    
    if USE_CHATGPT:
        reply = ask_chatgpt(text, mood)
    else:
        reply = ask_llama(text, mood)

    print(f"🤖 AI: {reply}")
    speak(reply)


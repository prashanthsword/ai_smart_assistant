import cv2
import torch
import pyttsx3
import subprocess
import speech_recognition as sr
from fer import FER
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import sounddevice as sd
import warnings
warnings.filterwarnings("ignore")


# === INIT SETTINGS ===
name = "Prashanth"
emotion_model = FER(mtcnn=True)
engine = pyttsx3.init()

# === TEXT-TO-SPEECH ===
def speak(text):
    print(f"AI ({detected_mood}):", text)
    engine.say(text)
    engine.runAndWait()

# === VOICE INPUT ===
def record_voice(seconds=5, fs=16000):
    print("🎙️ Speak now...")
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return torch.tensor(audio.flatten())

# === MOOD DETECTION (Voice) ===
def detect_mood_from_voice(audio_tensor):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    input_values = processor(audio_tensor, return_tensors="pt", sampling_rate=16000).input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0].lower()

    return transcription

# === MOOD DETECTION (Face) ===
def detect_mood_from_face():
    cap = cv2.VideoCapture(0)
    detected = "neutral"
    print("📸 Detecting facial mood...")

    for _ in range(30):  # Check ~3 seconds
        ret, frame = cap.read()
        if not ret:
            continue
        result = emotion_model.top_emotion(frame)
        if result:
            emotion, score = result
            detected = emotion
            cv2.putText(frame, f"{emotion} ({round(score * 100)}%)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Mood Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return detected

# === ASK LLaMA3 ===
def ask_llama3(prompt):
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    output = result.stdout.decode().split(">>>")[-1].strip()
    return output

# === MAIN LOGIC ===
print(f"👋 Hello {name}, I'm your AI assistant!")
detected_mood = detect_mood_from_face()
speak(f"Hi {name}, I see you're feeling {detected_mood} today. Let's begin!")

while True:
    try:
        audio_tensor = record_voice()
        query = detect_mood_from_voice(audio_tensor)

        if any(word in query.lower() for word in ["exit", "quit", "bye"]):
            speak("Goodbye! See you later.")
            break

        mood_prefix = f"[Detected Mood: {detected_mood}]\nUser said: {query}"
        reply = ask_llama3(mood_prefix)
        speak(reply)

    except KeyboardInterrupt:
        speak("Goodbye!")
        break
    except Exception as e:
        speak("Sorry, something went wrong.")
        print(f"❌ Error: {e}")

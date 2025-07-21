import subprocess
import pyttsx3
import speech_recognition as sr

# Initialize text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    print(f"🗣️ AI says: {text}")
    engine.say(text)
    engine.runAndWait()

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Listening...")
        audio = recognizer.listen(source)

    try:
        print("🔍 Recognizing...")
        query = recognizer.recognize_google(audio)
        print(f"You said: {query}")
        return query
    except sr.UnknownValueError:
        print("❌ Sorry, I couldn't understand.")
        return ""
    except sr.RequestError:
        print("❌ Could not request results.")
        return ""

def ask_llama(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3"],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        output = result.stdout.decode().strip()
        # Clean response
        if ">>>" in output:
            return output.split(">>>")[-1].strip()
        return output
    except Exception as e:
        return f"Error: {str(e)}"

# 🔁 Main loop
print("🤖 LLaMA 3 Voice Assistant is ready! Say 'exit' to stop.")
while True:
    user_input = listen()
    if user_input.lower() in ["exit", "quit", "bye"]:
        speak("Goodbye! See you next time.")
        break

    if user_input.strip() == "":
        continue

    reply = ask_llama(user_input)
    speak(reply)

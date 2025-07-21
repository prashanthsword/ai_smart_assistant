import speech_recognition as sr
import subprocess

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
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    response = result.stdout.decode().strip().split("\n")[-1]
    return response

# Main loop
print("🤖 Your AI Assistant is ready! Speak into the mic.")
while True:
    user_input = listen()
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("👋 Exiting. See you later!")
        break

    if user_input.strip() == "":
        continue

    reply = ask_llama(user_input)
    print(f"AI: {reply}")

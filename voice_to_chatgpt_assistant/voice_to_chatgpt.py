import openai
import pyttsx3
import speech_recognition as sr 

openai.api_key = "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"

engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        audio = r.listen(source)
    try:
        print("🔎 Recognizing...")
        query = r.recognize_google(audio)
        print(f"You said: {query}")
        return query
    except sr.UnknownValueError:
        print("😕 Sorry, I didn't catch that.")
        return ""
    except sr.RequestError:
        print("⚠️ Speech recognition error.")
        return ""

def ask_chatgpt(question):
    print("🤖 Thinking...")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful smart assistant."},
            {"role": "user", "content": question}
        ]
    )
    answer = response.choices[0].message.content
    print(f"ChatGPT: {answer}")
    return answer
if __name__ == "__main__":
    while True:
        query = listen()
        if query.lower() in ["exit", "quit", "bye"]:
            speak("Goodbye!")
            break
        if query:
            reply = ask_chatgpt(query)
            speak(reply)

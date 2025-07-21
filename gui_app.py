import streamlit as st
from utils import multi_modal_mood_detection
import llama_voice
import llama_chat

# Set Streamlit page
st.set_page_config(page_title="AI Smart Assistant", layout="centered")
st.title("🤖 AI Smart Assistant")

# --- Mood Detection Button ---
if st.button("😎 Detect Mood (face + voice)"):
    st.info("Running mood detection...")
    mood = multi_modal_mood_detection.detect_mood()
    st.success(f"Detected Mood: {mood}")

# --- Voice Assistant Button ---
if st.button("🎤 Start Voice Assistant"):
    with st.spinner("Listening..."):
        response = llama_voice.run_voice_assistant()
    st.success(f"🧠 Assistant: {response}")

# --- Text Input and Chat ---
st.markdown("### 💬 Chat with Assistant")
user_input = st.text_input("Type your message:")

if st.button("Send"):
    if user_input:
        with st.spinner("Thinking..."):
            reply = llama_chat.chat_with_llama(user_input)  # You'll define this
        st.success(f"🤖 Assistant: {reply}")
    else:
        st.warning("Please type a message.")

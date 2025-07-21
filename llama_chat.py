import subprocess

def ask_llama3(prompt):
    # Run the prompt with ollama
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return result.stdout.decode("utf-8")

print("🤖 Your Local AI Assistant is Ready! Type 'exit' to quit.")

while True:
    try:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        print("AI:", ask_llama3(user_input).split(">>>")[-1].strip())

    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        break
    except Exception as e:
        print(f"❌ Error: {e}")

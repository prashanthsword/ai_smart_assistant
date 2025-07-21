import cv2
from deepface import DeepFace
from transformers import pipeline

emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
result = emotion_classifier("I'm feeling really anxious today.")
print(result)

def detect_mood():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not accessible.")
        return

    print("📸 Looking at your face... (Press 'q' to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Analyze the current frame for emotion
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']
            print(f"🧠 Detected Emotion: {emotion}")

            # Display the video feed with emotion
            cv2.putText(frame, f"Mood: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        except Exception as e:
            print("⚠️ Face not detected clearly")

        cv2.imshow("Mood Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_mood()

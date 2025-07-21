import cv2
from fer import FER

# Initialize video capture
cap = cv2.VideoCapture(0)
detector = FER(mtcnn=True)

print("📸 Real-time Mood Detection - Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Detect emotion
    results = detector.detect_emotions(frame)
    for result in results:
        (x, y, w, h) = result["box"]
        emotion, score = max(result["emotions"].items(), key=lambda item: item[1])
        
        # Draw box and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame, f"{emotion} ({int(score * 100)}%)",
            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

    cv2.imshow("Mood Detection - Press 'q' to quit", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting mood detector...")
        break

cap.release()
cv2.destroyAllWindows()

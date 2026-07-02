import cv2
import mediapipe as mp

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Fingertip landmark IDs
tip_ids = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()

    if not success:
        break

    # Mirror the image
    img = cv2.flip(img, 1)

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process image
    results = hands.process(img_rgb)

    finger_count = 0

    if results.multi_hand_landmarks and results.multi_handedness:

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            # Left or Right hand
            hand_label = results.multi_handedness[idx].classification[0].label

            landmarks = []

            h, w, _ = img.shape

            # Save landmark coordinates
            for lm in hand_landmarks.landmark:
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                landmarks.append((cx, cy))

            # -------------------------
            # Thumb
            # -------------------------
            if hand_label == "Right":
                if landmarks[4][0] > landmarks[3][0]:
                    finger_count += 1
            else:  # Left hand
                if landmarks[4][0] < landmarks[3][0]:
                    finger_count += 1

            # -------------------------
            # Other four fingers
            # -------------------------
            for tip in tip_ids[1:]:
                if landmarks[tip][1] < landmarks[tip - 2][1]:
                    finger_count += 1

            # Draw landmarks
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Display Left/Right label
            cv2.putText(
                img,
                hand_label,
                (landmarks[0][0] - 20, landmarks[0][1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2
            )

    # Display total finger count
    cv2.rectangle(img, (20, 20), (260, 120), (0, 255, 0), -1)

    cv2.putText(
        img,
        f"Fingers: {finger_count}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        3
    )

    cv2.imshow("Finger Counter", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
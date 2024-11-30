import cv2
import mediapipe as mp
import numpy as np
import math
from telegram import Bot
import tracemalloc
import requests

tracemalloc.start()

telegram_bot_token = "7156811542:AAHuK_d-njPwXjBz8k9M25EWax0JnbX4l7A"
bot = Bot(token=telegram_bot_token)
chat_id = "1869333272" 

def calculate_angle(a, b, c):
    radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle = abs(math.degrees(radians))
    return angle


def send_telegram_message(message):
    send_text = "https://api.telegram.org/bot" + telegram_bot_token + "/sendMessage?chat_id=" + chat_id + "&text=" + message
    response = requests.get(send_text)
    return response.json()


def is_fallen(left_shoulder, left_hip, left_knee, left_ankle, right_shoulder, right_hip, right_knee, right_ankle):
    angle_left_shoulder_hip_knee = calculate_angle(left_shoulder, left_hip, left_knee)
    angle_left_hip_knee_ankle = calculate_angle(left_hip, left_knee, left_ankle)
    angle_right_shoulder_hip_knee = calculate_angle(right_shoulder, right_hip, right_knee)
    angle_right_hip_knee_ankle = calculate_angle(right_hip, right_knee, right_ankle)

    shoulder_hip_knee_threshold = 120
    hip_knee_ankle_threshold = 120

    left_side_fallen = angle_left_shoulder_hip_knee > shoulder_hip_knee_threshold and angle_left_hip_knee_ankle > hip_knee_ankle_threshold
    right_side_fallen = angle_right_shoulder_hip_knee > shoulder_hip_knee_threshold and angle_right_hip_knee_ankle > hip_knee_ankle_threshold

    return left_side_fallen or right_side_fallen

def main():
    cap = cv2.VideoCapture(0)

    frame_count = 0

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Load pre-trained MobileNet SSD for human detection
    net = cv2.dnn.readNetFromCaffe('models/MobileNetSSD_deploy.prototxt', 'models/MobileNetSSD_deploy.caffemodel')

    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:

        while cap.isOpened():

            print(f"Frame {frame_count} Processing")
            frame_count += 1
        
            ret, frame = cap.read()
            if not ret:
                break

            # Human detection
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.9:  # Adjust confidence threshold as needed
                    box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Calculate length and breadth
                    length = endY - startY
                    breadth = endX - startX

                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Pose detection
            results = pose.process(frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

                if is_fallen(left_shoulder, left_hip, left_knee, left_ankle, right_shoulder, right_hip, right_knee, right_ankle):
                    if(length - breadth < 0):
                        cv2.putText(frame, "Fall Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        send_telegram_message("Someone Fell Down!(image)")

            cv2.imshow("Fall Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

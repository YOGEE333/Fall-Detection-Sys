import cv2
import mediapipe as mp
import math

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Function to check if a person is sitting
def is_person_sitting(left_ankle, right_ankle, left_knee, right_knee, left_hip, right_hip, left_shoulder, right_shoulder):
    
    # Calculate angles
    angle_left_knee = calculate_angle(left_hip, left_knee, left_ankle)
    angle_right_knee = calculate_angle(right_hip, right_knee, right_ankle)
    angle_left_hip = calculate_angle(left_knee, left_hip, left_shoulder)
    angle_right_hip = calculate_angle(right_knee, right_hip, right_shoulder)

    threshold_angle = 90
    # Check if both angles are less than threshold_angle degrees (sitting position)
    if angle_left_knee < threshold_angle  and angle_right_knee < threshold_angle and angle_right_hip > threshold_angle and angle_left_hip > threshold_angle:
        return True
    return False

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize video capture
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image with MediaPipe Pose model
        results = pose.process(image_rgb)

        # Extract pose landmarks
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]


            # Check if a person is sitting based on the detected landmarks
            if is_person_sitting(left_ankle, right_ankle, left_knee, right_knee, left_hip, right_hip, left_shoulder, right_shoulder):
                cv2.putText(image, "Sitting", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(image, "Not Sitting", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Render the result
        cv2.imshow('MediaPipe Pose Detection', image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

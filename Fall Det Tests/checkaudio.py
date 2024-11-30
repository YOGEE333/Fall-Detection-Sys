import mediapipe as mp
import cv2

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to find the center of the body
def find_body_center(image):
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run MediaPipe Pose model on the image
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        # Get landmarks of the body
        landmarks = results.pose_landmarks.landmark
        
        # Calculate the center of the body based on specific landmarks
        # For simplicity, let's assume center is the midpoint between the left and right hip
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        center_x = int((left_hip.x + right_hip.x) / 2 * image.shape[1])
        center_y = int((left_hip.y + right_hip.y) / 2 * image.shape[0])
        
        return center_x, center_y
    else:
        return None

# Open a video capture device (here assumed to be the first camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break
    
    # Find center of the body
    center = find_body_center(frame)

    if center:
        # Draw a circle at the center of the body
        cv2.circle(frame, center, 5, (0, 255, 0), -1)
    
    # Display the frame
    cv2.imshow('Body Center Detection', frame)
    
    # Check for 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pose.close()

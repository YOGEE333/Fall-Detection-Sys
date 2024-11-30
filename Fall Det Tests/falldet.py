import cv2
import mediapipe as mp
import numpy as np
import math
import speech_recognition as sr
import pyttsx3
from telegram import Bot
import subprocess
import tracemalloc
import requests
import time

tracemalloc.start()

recognizer = sr.Recognizer()
engine = pyttsx3.init()

telegram_bot_token = "7156811542:AAHuK_d-njPwXjBz8k9M25EWax0JnbX4l7A"
bot = Bot(token=telegram_bot_token)
chat_id = "1869333272" 

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle = abs(math.degrees(radians))
    return int(angle)


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


def listen():
    with sr.Microphone() as source:
        print("Listening for 'help'...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        query = recognizer.recognize_google(audio).lower()
        print("You said:", query)
        return query
    
    except sr.UnknownValueError:
        print("Sorry, I didn't catch that.")
        return ""
    

def send_telegram_message(message):
    send_text = "https://api.telegram.org/bot" + telegram_bot_token + "/sendMessage?chat_id=" + chat_id + "&text=" + message
    response = requests.get(send_text)
    return response.json()


def send_image(imagePath):
    command = 'curl -s -X POST https://api.telegram.org/bot' + telegram_bot_token + '/sendPhoto -F chat_id=' + chat_id + " -F photo=@" + imagePath
    subprocess.call(command.split(' '))    
    return


def fall_det(imagex):

    cap = imagex
    
    frame_count = 1

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Load pre-trained MobileNet SSD for human detection

    net = cv2.dnn.readNetFromCaffe('models/MobileNetSSD_deploy.prototxt', 'models/MobileNetSSD_deploy.caffemodel')

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():

            print(f"Frame {frame_count} Processing")
            frame_count += 1

            ret, frame = cap.read()
            
            frame = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_AREA)
            
            if not ret:
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_height ,image_width ,_ = image.shape 

            # Human detection
            blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.4:  # Adjust confidence threshold as needed
                    box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Calculate length and breadth
                    length = endY - startY
                    breadth = endX - startX

                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)


            
            
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            #try:
            if results.pose_landmarks:

                landmarks = results.pose_landmarks.landmark
                
                # ----------------------   DOT   ----------------------           
                
                # dot - LEFT_HIP
                    
                dot_LEFT_HIP_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image_width)
                dot_LEFT_HIP_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image_height)
                
                # dot - RIGHT_HIP
                    
                dot_RIGHT_HIP_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width)
                dot_RIGHT_HIP_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image_height)
                
                # dot - LEFT_KNEE
                    
                dot_LEFT_KNEE_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * image_width)
                dot_LEFT_KNEE_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * image_height)
                            
                # dot - RIGHT_KNEE
                    
                dot_RIGHT_KNEE_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * image_width)
                dot_RIGHT_KNEE_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * image_height)
                
                # dot - LEFT_HEEL
                    
                dot_LEFT_HEEL_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].x * image_width)
                dot_LEFT_HEEL_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].y * image_height)
            
                # dot - RIGHT_HEEL
                    
                dot_RIGHT_HEEL_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].x * image_width)
                dot_RIGHT_HEEL_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].y * image_height)
                
                # dot - LEFT_FOOT_INDEX
                    
                dot_LEFT_FOOT_INDEX_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * image_width)
                dot_LEFT_FOOT_INDEX_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * image_height)
            
                # dot - RIGHT_FOOT_INDEX
                    
                dot_RIGHT_FOOT_INDEX_X= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * image_width)
                dot_RIGHT_FOOT_INDEX_Y= int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * image_height)
                
                # dot _ UPPER_BODY
                
                dot_UPPER_BODY_X = int( (dot_LEFT_HIP_X + dot_RIGHT_HIP_X)/2 )
                dot_UPPER_BODY_Y = int( (dot_LEFT_HIP_Y + dot_RIGHT_HIP_Y)/2 )
                
                # dot _ LOWER_BODY
                
                dot_LOWER_BODY_X = int( (dot_LEFT_KNEE_X + dot_RIGHT_KNEE_X)/2 )
                dot_LOWER_BODY_Y = int( (dot_LEFT_KNEE_Y + dot_RIGHT_KNEE_Y)/2 )
                
                # dot _ BODY
                
                dot_BODY_X = int( (dot_UPPER_BODY_X + dot_LOWER_BODY_X)/2 )
                dot_BODY_Y = int( (dot_UPPER_BODY_Y + dot_LOWER_BODY_Y)/2 )
            
                #for feet
                Point_of_action_LEFT_X = int( 
                    ((dot_LEFT_FOOT_INDEX_X +  dot_LEFT_HEEL_X)/2) )
                
                Point_of_action_LEFT_Y = int( 
                    ((dot_LEFT_FOOT_INDEX_Y+   dot_LEFT_HEEL_Y)/2) )
                
                Point_of_action_RIGHT_X = int( 
                    ((dot_RIGHT_FOOT_INDEX_X +  dot_RIGHT_HEEL_X)/2) )
                
                Point_of_action_RIGHT_Y = int( 
                    ((dot_RIGHT_FOOT_INDEX_Y+   dot_RIGHT_HEEL_Y)/2) )           
                
                #co ords between feet
            
                Point_of_action_X = int ( (Point_of_action_LEFT_X +  Point_of_action_RIGHT_X)/2 )
                
                Point_of_action_Y = int ( (Point_of_action_LEFT_Y +  Point_of_action_RIGHT_Y)/2 )
                

                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                
                #fall case
                fall = int(Point_of_action_X - dot_BODY_X )

                #case falling and standing
            
                falling = abs(fall) > 50
                #standing = abs(fall) < 50
                
                #x = Point_of_action_X
                #y = -(1.25*x) + 618
                flag = 0

                if falling:
                    if is_fallen(left_shoulder, left_hip, left_knee, left_ankle, right_shoulder, right_hip, right_knee, right_ankle):
                        if(length - breadth < 0):
                            flag = 1
                            cv2.putText(image, "Fall Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            send_telegram_message("Someone Fell Down!(image)")
                            print("message sent")
                            #send_image(image)


                # elif Point_of_action_X <  320 and Point_of_action_X > 100 and  Point_of_action_Y > 390 and Point_of_action_Y > y and  standing :
                #     if is_fallen(left_shoulder, left_hip, left_knee, left_ankle, right_shoulder, right_hip, right_knee, right_ankle):
                #         if(length - breadth < 0):
                #             cv2.putText(frame, "Person Fall Detected!", (11, 100), 0, 1, [0, 0, 2550],thickness=3, lineType=cv2.LINE_AA)
                #             send_telegram_message("Someone Fell Down (image)!")
                #             send_image(image)
                            

                # elif Point_of_action_X >=  320 and Point_of_action_Y > 240 and standing :
                #     if is_fallen(left_shoulder, left_hip, left_knee, left_ankle, right_shoulder, right_hip, right_knee, right_ankle):
                #         if(length - breadth < 0):
                #             cv2.putText(frame, "Person Fall Detected!", (11, 100), 0, 1, [0, 0, 2550],thickness=3, lineType=cv2.LINE_AA)
                #             send_telegram_message("Someone Fell Down!(image)")
                #             send_image(image)
                                                
        #except:
            #pass

            cv2.imshow('Media pipe Feed', image)
    
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    image  = cv2.VideoCapture("../fdx2/dataset/FALL-Backwards_.mp4")

    fall_det(image)
    # while True:
    #     query = listen()
    #     if "help" in query:
    #         send_telegram_message("Someone Fell Down(audio)!")
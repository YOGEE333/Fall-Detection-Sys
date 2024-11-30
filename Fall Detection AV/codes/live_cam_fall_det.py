#imports for the build

import cv2                                     #camera use
import mediapipe as mp                         #mediapipe for human body recognition and landmarks
import numpy as np                             #for arrays of points of vertices of the human body box
import math                                    #for calculating angles
import speech_recognition as sr                #for speech recognition
import pyttsx3                                 #for microphone driver access
from telegram import Bot                       #for telegram bot
import subprocess                              #to run other program or command  
import tracemalloc                             #for memory block access by python 
import requests                                #for generating api requests for telegram messages
import threading                               #for multithreading to run multiple processes together

tracemalloc.start()

recognizer = sr.Recognizer()   #speech recognition initialized
engine = pyttsx3.init()        #microphone initialized

#variables declared for telegram bot

telegram_bot_token = "7156811542:AAHuK_d-njPwXjBz8k9M25EWax0JnbX4l7A"
bot = Bot(token=telegram_bot_token)
chat_id = "1869333272" 

#mediapipe utils for landmark extraction

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#function to calculate angle between three points on human body recognized by mediapipe

def calculate_angle(a, b, c):
    radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle = abs(math.degrees(radians))
    return int(angle)

#function to consider a fall based on the posture of human body 

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

#function for speech recognition

def listen(image):
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
    
#function to generate a text message

def send_telegram_message(message):
    send_text = "https://api.telegram.org/bot" + telegram_bot_token + "/sendMessage?chat_id=" + chat_id + "&text=" + message
    response = requests.get(send_text)
    return response.json()

#function to send an image as a message 

def send_image(imagePath):
    command = 'curl -s -X POST https://api.telegram.org/bot' + telegram_bot_token + '/sendPhoto -F chat_id=' + chat_id + " -F photo=@" + imagePath
    subprocess.call(command.split(' '))    
    return

#function to detect fall on the basis of center of mass of the human body 

def fall_det(video_capture):

    cap = video_capture

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
                
                #co ordinates between feet
            
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

                #condition for a fall
                falling = abs(fall) > 50

                #all the functions are used together to consider a fall
                if falling:
                    if is_fallen(left_shoulder, left_hip, left_knee, left_ankle, right_shoulder, right_hip, right_knee, right_ankle):
                        if(length - breadth < 0):
                            flag = 1
                            cv2.putText(image, "Fall Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.imwrite('fall.jpg',image)
                            print("message sent")
                            send_image('fall.jpg')

            cv2.imshow('Media pipe Feed', image)
    
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

#run_audio_listener function to check for the keyword "help" in the recognized text of speech

def run_audio_listener(video_capture):
    while True:
        query = listen(video_capture)
        if "help" in query:
            send_telegram_message("Someone needs Help!!!!")

#main to run both the features together using different functions

if __name__ == "__main__":

    video_capture = cv2.VideoCapture(0)
    video_thread = threading.Thread(target=fall_det,args=(video_capture,))
    video_thread.start()

    audio_thread = threading.Thread(target=run_audio_listener, args=(video_capture,))
    audio_thread.start()
    
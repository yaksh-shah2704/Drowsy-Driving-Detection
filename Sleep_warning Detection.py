import cv2
import time

import imutils
# face_units for basic operations of coversation
from imutils import face_utils

# Dlib for deep learning based Modules and face landmark detection
import dlib
from pygame import mixer
from scipy.spatial import distance

# load the sound which will play when the user is drowsy
mixer.init()
beep = mixer.Sound("beep.mp3")
# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A+B)/(2.0*C)     # Euclidean Distance
    return ear

# Threshold for drowsiness detection
thresh = 0.25
frame_check = 10        # Number of frames for drowsiness check
flag = 0

# Indices for the left and right eye landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']

# Load dlib's face detector and shape predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")     # it detects 68 landmmarks from our face like eye, lips etc.

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width = 550)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    
    # Detect faces in the grayscale image
    subjects = detect(gray, 0)
    
    for subject in subjects:
        # Get the landmarks for each face detected
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)       # Convert landmarks to numpy array
        
        # Get the left and right eye landmarks
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        # Calculate the eye aspect ratio for each eye
        leftEar = eye_aspect_ratio(leftEye)
        rightEar = eye_aspect_ratio(rightEye)
        
        # Calculate the average EAR
        ear = (leftEar + rightEar)/2.0
        
        # Draw the convex hull around both eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # Check if the EAR is below the threshold for drowsiness
        if ear<thresh:
            flag += 1
            print(flag)
            if flag >= frame_check:
                cv2.putText(frame, "*****Alert!!!*****", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "*****Wake up!!!*****", (10, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # mixer.music.play()
                # Play and pause the sound in intervals
                beep.play()
                
        else:
            flag = 0    # Reset the flag if eyes are open
            mixer.music.stop()
                
    # Display the frame with the detected landmarks and alert           
    cv2.imshow("Frame", frame)
    
    # # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
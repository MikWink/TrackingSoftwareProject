# Python 3.10 is required. Python 3.11 is not recommended/not working
# Imports needed are OpenCV, Mediapipe, NumPy, Python-OSC
# These Libraries need to be installed first, using following commands
# pip install mediapipe
# pip install opencv
# pip install numpy
# pip install python-osc

import mediapipe as mp
import cv2
import numpy as np
from pythonosc.udp_client import SimpleUDPClient

#############
### Setup ###
#############

# Aliases for Mediapipe objects
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Choosing font for display with OpenCV
font = cv2.FONT_HERSHEY_SIMPLEX

# Setup IP and Port for OSC transmission
ip = "192.168.1.208"
port = 1234

# Initialize UDPClient to send OSC Messages
client = SimpleUDPClient(ip, port)


# Initialize webcam feed
cap = cv2.VideoCapture(0)
# Set the width, height and FPS
camWidth = 1280
camHeight = 720
fps = 30
cap.set(3, camWidth)
cap.set(4, camHeight)
cap.set(5, fps)

# Create array holding the points to base the transformation of the screen on
points = []

# Colors for hands
colors = [(250, 44, 250), (250, 44, 0), (0, 44, 250)]

#################
### Functions ###
#################
def drawRect (event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

####################
### Program loop ###
####################

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=4) as hands:
    while cap.isOpened():
        
        ret, frame = cap.read()
        if ret:
            if len(points) == 4:
                src_points = []
                dst_points = []
                for i in range(4):
                    src_points.append((points[i][0], points[i][1]))
                    dst_points.append((0 if i == 0 or i == 3 else camWidth, 0 if i < 2 else camHeight))

                transform, _ = cv2.findHomography(np.array(src_points), np.array(dst_points), cv2.RANSAC)
                frame = cv2.warpPerspective(frame, transform, (camWidth, camHeight))

            # BGR 2 RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Set flag
            image.flags.writeable = False
            # Detections
            results = hands.process(image)
            # Set flag to true
            image.flags.writeable = True
            # RGB 2 BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            tolerance = 0.03
            # Are there any Hands found?
            if results.multi_hand_landmarks:
                # For-Loop to cycle through all the found hands
                for num, hand in enumerate(results.multi_hand_landmarks):
                    # Draw Hands with different color
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=colors[num], thickness=2, circle_radius=2))

                    # Save coordinates of the tip of the index finger
                    indexPos = [0,0]
                    indexPos[0] = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                    indexPos[1] = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x

                    # Save coordinates of the tip if the thumb
                    thumbX = hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x
                    thumbY = hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y

                    # Recognition of click gesture
                    if indexPos[0] - tolerance <= thumbX <= indexPos[0] + tolerance \
                            and indexPos[1] - tolerance <= thumbY <= indexPos[1] + tolerance:
                        cv2.putText(image, 'Click', (10, 90), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        client.send_message("/click", True)

                    client.send_message("/boardgame", indexPos)

            cv2.imshow("Webcam Feed", image)
            key = cv2.waitKey(1) & 0xFF
            cv2.setMouseCallback("Webcam Feed", drawRect)
            if key == ord('q'):
                break
            elif key == ord('c'):
                points = []
            elif key == ord('p'):
                cv2.imwrite("bild.jpg", image)

    cap.release()
    cv2.destroyAllWindows()

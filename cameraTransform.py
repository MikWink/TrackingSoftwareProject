import time
import mediapipe as mp
import cv2
import numpy as np
from pythonosc.udp_client import SimpleUDPClient

#############
### Setup ###
#############

camWidth = 1280
camHeight = 720

points2 = []

playerOSCAddress = "/0"

# Aliases for Mediapipe objects
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Choosing font for display with OpenCV
font = cv2.FONT_HERSHEY_SIMPLEX

def setup_webcam():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    fps = 60
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camHeight)
    cv2.namedWindow("Webcam Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Webcam Feed", camWidth, camHeight)
    cap.set(cv2.CAP_PROP_FORMAT, 0)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    return cap

def setup_osc_client(ip, port):
    return SimpleUDPClient(ip, port)

def initialize_hands():
    return mp.solutions.hands.Hands(
        min_detection_confidence=0.8,
        min_tracking_confidence=0.5,
        max_num_hands=4,
        model_complexity=0
    )

def draw_rect(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Point saved...")
        param.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        param.clear()
        print("Points deleted...")

def check_player_position(x, y, address):
    if 0 <= x <= 0.4 and 0 <= y <= 1:
        return address + "1"
    elif 0.6 <= x <= 1 and 0 <= y <= 1:
        return address + "2"
    elif 0.4 <= x <= 0.6 and 0 <= y <= 0.5:
        return address + "3"
    elif 0.4 <= x <= 0.6 and 0.5 <= y <= 1:
        return address + "4"

def process_frame(frame, hands, client, points, prev_frame_time):
    colors = [(250, 44, 250), (250, 44, 0), (0, 44, 250), (255, 44, 255)]
    click = False
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    #image = cv2.resize(image, (int(800*16/9), 800))
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    tolerance = 0.03

    if results.multi_hand_landmarks:
        for num, hand in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=colors[num], thickness=2, circle_radius=2))

            # Save coordinates of the tip of the index finger
            indexPos = [0, 0]
            indexPos[0] = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
            indexPos[1] = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

            # Save coordinates of the tip if the thumb
            thumbX = hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x
            thumbY = hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y

             # Recognition of click gesture
            if indexPos[0] - tolerance <= thumbX <= indexPos[0] + tolerance and indexPos[1] - tolerance <= thumbY <= indexPos[1] + tolerance:
                cv2.putText(image, 'Click', (10, 90), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                click = True
            else:
                click = False

            playerOSCAddress = check_player_position(indexPos[0], indexPos[1], "/player")
            OSCAddress = "/player" + str(num + 1)
            client.send_message(playerOSCAddress, indexPos)
            playerOSCAddress = check_player_position(indexPos[0], indexPos[1], "/click")
            OSCAddress = "/click" + str(num + 1)
            client.send_message(playerOSCAddress, click)
            print(str(playerOSCAddress) + ": " + str(indexPos[0]) + ", " + str(indexPos[1]) + ", " + str(click))

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    cv2.putText(image, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Webcam Feed", image)
    return cv2.waitKey(10) & 0xFF

#####################
### Program Start ###
#####################

def main():
    # Constants
    ip = "127.0.0.1"
    port = 1234

    # Initialize components
    cap = setup_webcam()
    client = setup_osc_client(ip, port)
    hands = initialize_hands()
    

    points2 = []
    cv2.setMouseCallback("Webcam Feed", draw_rect, points2)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if len(points2) == 4:
                src_points = []
                dst_points = []
                for i in range(4):
                    src_points.append((points2[i][0], points2[i][1]))
                    dst_points.append((0 if i == 0 or i == 3 else camWidth, 0 if i < 2 else camHeight))
                transform, _ = cv2.findHomography(np.array(src_points), np.array(dst_points), cv2.RANSAC)
                frame = cv2.warpPerspective(frame, transform, (camWidth, camHeight))

            #frame = cv2.flip(frame, 1)
            prev_frame_time = time.time()
            key = process_frame(frame, hands, client, points2, prev_frame_time)
            
            
            if key == ord('q'):
                break
            elif key == ord('p'):
                cv2.imwrite("bild.jpg", frame)
                print("Picture taken...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

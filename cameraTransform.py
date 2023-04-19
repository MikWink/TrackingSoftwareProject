import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
from pythonosc.udp_client import SimpleUDPClient

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

font = cv2.FONT_HERSHEY_SIMPLEX

ip = "192.168.1.208"
port = 1234

#Test commit

# Initialize webcam feed
cap = cv2.VideoCapture(0)
# Set the frame width and height
cap.set(3, 1280)
cap.set(4, 720)
# Set the frames per second (fps)
cap.set(5, 30)

game = "shark"
gameList = "boardgame", "shark", "pong", "puzzle"
gameIndex = 0

camWidth = 1280
camHeight = 720

points = []

def drawRect (event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

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

            # Detections
            # print(results)
            tolerance = 0.03
            indexY = 0
            # Rendering results
            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))

                    indexY = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                    indexX = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x

                    thumbX = hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x
                    thumbY = hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y

                    client = SimpleUDPClient(ip, port)  # Create client

                    if thumbX >= indexX - tolerance and thumbX <= indexX + tolerance and thumbY >= indexY - tolerance and thumbY <= indexY + tolerance:
                        cv2.putText(image, 'Click', (10, 90), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        client.send_message("/click", True)
                    playerIndex = 0
                    indexPosInd = [playerIndex, indexX, indexY]
                    indexPos = [indexX, indexY]

                    # click gesture
                    if game == "pong":
                        if indexX >= 0.5:
                            client.send_message("/player1Pong", indexY)  # Send float message
                            cv2.putText(image, '{0:.2f}'.format(indexY), (10, 50), font, 1, (255, 255, 255), 2,
                                        cv2.LINE_AA)
                        else:
                            client.send_message("/player2Pong", indexY)
                            cv2.putText(image, '{0:.2f}'.format(indexY), (560, 50), font, 1, (255, 255, 255), 2,
                                        cv2.LINE_AA)
                    elif game == "shark":
                        client.send_message("/sharkX", indexX)
                        client.send_message("/sharkY", indexY)
                    elif game == "puzzle":
                        if indexX < 0.45 and indexY < 0.45:
                            indexPosInd[0] = 0
                            client.send_message("/player1", indexPosInd)

                        elif indexX > 0.55 and indexY < 0.45:
                            indexPosInd[0] = 1
                            client.send_message("/player2", indexPosInd)
                        elif indexX < 0.45 and indexY > 0.55:
                            indexPosInd[0] = 2
                            client.send_message("/player3", indexPosInd)
                        elif indexX > 0.55 and indexY > 0.55:
                            indexPosInd[0] = 3
                            client.send_message("/player4", indexPosInd)
                    elif game == "boardgame":
                        client.send_message("/boardgame", indexPos)
                        print("(DEBUG, OSC) Out: {")
                        print("address: /boardgame")
                        print("args: [{type: f, value: ", indexPos[0], "}],")
                        print("      [{type: f, value: ", indexPos[1], "}]")
                        print("}")
                        print("To: 192.168.178.41:8080")

            cv2.imshow("Webcam Feed", image)
            key = cv2.waitKey(1) & 0xFF
            cv2.setMouseCallback("Webcam Feed", drawRect)
            if key == ord('q'):
                break
            elif key == ord('c'):
                points = []
            elif key == ord('g'):
                if gameIndex < len(gameList) - 1:
                    gameIndex += 1
                else:
                    gameIndex = 0
                game = gameList[gameIndex]
                print(game)
            elif key == ord('p'):
                cv2.imwrite("bild.jpg", image)



cap.release()
cv2.destroyAllWindows()
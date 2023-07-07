import time
import mediapipe as mp
import cv2
import numpy as np
from pythonosc.udp_client import SimpleUDPClient

def detect_pose_landmarks(pose, frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    return results.pose_landmarks

def show_webcam_feed(window_name, frame):
    cv2.imshow(window_name, frame)

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


    
def map_coordinates_x(x):    
    result =  (x- 100/ 1280) * 1/(1080/1280) 
    if result < 0:
        return 0
    elif result > 1:
        return 1
    else:
        return result

def map_coordinates_y(y):
    result = (y- 100/720) * 1/(520/720)
    if result < 0:
        return 0
    elif result > 1:
        return 1
    else:
        return result


def process_frame(frame, hands, client, points, prev_frame_time, hand_detected_time):
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
            print(num)
            #mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                      #mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      #amp_drawing.DrawingSpec(color=colors[num], thickness=2, circle_radius=2))

            # Save coordinates of the tip of the index finger
            indexPos = [0, 0]
            indexPos[0] = map_coordinates_x(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x)
            indexPos[1] = map_coordinates_y(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y)
            #print(indexPos)
            # Save coordinates of the tip if the thumb
            thumbX = hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x
            thumbY = hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y

             # Recognition of click gesture
            if indexPos[0] - tolerance <= thumbX <= indexPos[0] + tolerance and indexPos[1] - tolerance <= thumbY <= indexPos[1] + tolerance:
                cv2.putText(image, 'Click', (10, 90), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                click = True
            else:
                click = False
            OSCAddress = "/player" + str(num + 1)
            client.send_message(OSCAddress, indexPos)
            OSCAddress = "/click" + str(num + 1)
            client.send_message(OSCAddress, click)
            #print(str(playerOSCAddress) + ": " + str(indexPos[0]) + ", " + str(indexPos[1]) + ", " + str(click))

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    cv2.putText(image, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Webcam Feed", image)
    return results.multi_hand_landmarks

def do_the_handtracking(client):
    

    # Initialize components
    cap = setup_webcam()
    hands = initialize_hands()
    

    cv2.setMouseCallback("Webcam Feed", draw_rect, points2)
    hand_detected_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 0)
        if ret:
            if len(points2) == 4:
                src_points = []
                dst_points = []
                for i in range(4):
                    src_points.append((points2[i][0], points2[i][1]))
                    dst_points.append((100 if i == 0 or i == 3 else camWidth-100, 100 if i < 2 else camHeight-100))
                transform, _ = cv2.findHomography(np.array(src_points), np.array(dst_points), cv2.RANSAC)
                frame = cv2.warpPerspective(frame, transform, (camWidth, camHeight))
                for i in range(len(dst_points)):
                    cv2.line(frame, dst_points[i], dst_points[(i + 1) % len(dst_points)], (0, 255, 0), 2)

            
            prev_frame_time = time.time()
            if process_frame(frame, hands, client, points2, prev_frame_time, hand_detected_time):
                hand_detected_time = time.time()
            if (time.time() - hand_detected_time) > 10:
                break 
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(1) & 0xFF == ord('p'):
                cv2.imwrite("bild.jpg", frame)
                print("Picture taken...")

    cap.release()
    cv2.destroyWindow("Webcam Feed")

def main():
    # Constants
    ip = "192.168.178.56"
    port = 1234
    
    client = setup_osc_client(ip, port)
    mp_pose = mp.solutions.pose

    # Initialize cameras
    cap1 = cv2.VideoCapture(1)  # First camera
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Check if cameras are opened successfully
    if not cap1.isOpened():
        print("Error opening cameras.")
        return

    with mp_pose.Pose(min_detection_confidence=0.95, min_tracking_confidence=0.5) as pose:
        person_detected = False
        close_window1 = False
        close_window2 = False

        while True:
            # Read frames from both cameras
            ret1, frame1 = cap1.read()  # First camera

            if not ret1:
                break

            # Process the first frame to detect pose landmarks
            pose_landmarks = detect_pose_landmarks(pose, frame1)
            if pose_landmarks is not None:
                person_detected = True
                print("Person detected!")
            else:
                person_detected = False
                print("...")

            # Show the webcam feed only when a person is detected in the first camera
            if person_detected:
                client.send_message("/screen0", True)
                if close_window1:
                    cv2.destroyWindow('Human Detection')
                    close_window1 = False
                do_the_handtracking(client)
                close_window2 = True
            else:
                client.send_message("/screen1", True)
                if close_window2:
                    close_window2 = False
                show_webcam_feed('Human Detection', frame1)
                close_window1 = True

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            

    # Release the cameras and close windows
    cap1.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

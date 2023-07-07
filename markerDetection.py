import cv2
import numpy as np

points2 = []
camWidth = 1280
camHeight = 720


def draw_rectangle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Point saved...")
        param.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        param.clear()
        print("Points deleted...")



# Open the webcam
cap = cv2.VideoCapture(0)
fps = 60
cap.set(cv2.CAP_PROP_FPS, fps)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camHeight)
# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Failed to open webcam")
    exit()




while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if frame is captured successfully
    if not ret:
        print("Failed to capture frame")
        break
    if len(points2) == 4:
        src_points = []
        dst_points = []
        tracking_points = []
        for i in range(4):
            src_points.append((points2[i][0], points2[i][1]))
            dst_points.append((100 if i == 0 or i == 3 else camWidth -100, 100 if i < 2 else camHeight-100))
            tracking_points.append((0 if i == 0 or i == 3 else camWidth , 0 if i < 2 else camHeight))
        
        
        transform, _ = cv2.findHomography(np.array(src_points), np.array(dst_points), cv2.RANSAC)
        frame = cv2.warpPerspective(frame, transform, (camWidth, camHeight))
        for i in range(len(dst_points)):
            cv2.line(frame, dst_points[i], dst_points[(i + 1) % len(dst_points)], (0, 255, 0), 2)
    # Display the frame
    cv2.imshow("Webcam", frame)
    cv2.setMouseCallback("Webcam", draw_rectangle, points2)

    # Check for 'Esc' key press to exit
    if cv2.waitKey(1) == 27:
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

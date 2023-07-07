import cv2

# Define the list of coordinates (x, y) for the four points
points = [(100, 100), (200, 100), (200, 200), (100, 200)]

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame from the webcam
    ret, frame = cap.read()

    # Draw the lines connecting the points on the frame
    for i in range(len(points)):
        cv2.line(frame, points[i], points[(i + 1) % len(points)], (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Webcam Feed", frame)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()

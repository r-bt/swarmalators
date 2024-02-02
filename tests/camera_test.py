import cv2

def show_camera_feed():
    # Open the default camera (usually camera index 0)
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Display the frame
        cv2.imshow('Camera Feed', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start the live camera feed
show_camera_feed()

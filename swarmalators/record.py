import cv2
import time
import sys

record_index = int(sys.argv[1])

# Set up video capture
cap = cv2.VideoCapture(record_index)  # Change 0 to the appropriate camera index if needed
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up video writer using FFmpeg
current_time = time.strftime("%Y%m%d%H%M%S")
output_file = f"output_{current_time}.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change the codec as needed
out = cv2.VideoWriter(output_file, fourcc, 20.0, (width, height))  # Adjust parameters as needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the frame
    cv2.imshow('Recording...', frame)

    # Write the frame to the video file
    out.write(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
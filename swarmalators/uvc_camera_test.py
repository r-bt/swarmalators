from tracker import VideoStream, CameraSpec, CameraControls
import cv2
import time

camera = CameraSpec(
    uid="2:1",
    width=1920,
    height=1080,
    fps=30,
    bandwidth_factor=1.6,
)

stream = VideoStream(camera)

stream.start()

frame_time = 0
prev_frame_time = 0
while True:
    frame = stream.read()

    cv2.imshow('Camera Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


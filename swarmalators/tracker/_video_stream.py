import cv2
from threading import Thread
from .util._c930e import apply_camera_controls, CameraControls
import time
import numpy as np
from ._video_stream_pyuvc import CameraSpec
import subprocess

DEFAULT_CAMERA_CONTROLS = CameraControls(
    brightness=150,
    contrast=128,
    saturation=255,
    sharpness=255,
    zoom=104,
    gain=124,
    exposure_time=624,
    auto_focus=0,
    focus=0
)

class VideoStream:
    """A CV2 VideoStream wrapper for threading.

    Attributes:
        device: The device index of the camera to use.
    """

    def __init__(self, device: CameraSpec):
        self._stream = cv2.VideoCapture(0)

        if not self._stream.isOpened():
            self._stream.open(device)

        # Apply camera settings
        input_config = open("default_camera.json", "r")
        res = subprocess.run(["uvcc", "--product", "2115", "import"], stdin=input_config, capture_output=True, text=True) 
        input_config.close()
        print(res.stdout)
        print(res.stderr)
        # print("Applying camera controls")
        # apply_camera_controls(controls)
        # print("Applied camera controls")

        (self._grabbed, self._frame) = self._stream.read()

        self._stopped = False

    def start(self):
        """Start the thread to read frames from the video stream."""
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        """Keep looping indefinitely until the thread is stopped."""
        while True:
            if self._stopped:
                return
            
            try:
                (self._grabbed, self._frame) = self._stream.read()
            except:
                continue
    
    def read(self):
        """Return the current frame."""
        return self._frame.copy()
    
    def stop(self):
        """Indicate that the thread should be stopped."""
        self._stopped = True
        # Wait a moment to avoid segfaults
        time.sleep(0.5)
        # Release the stream
        self._stream.release()
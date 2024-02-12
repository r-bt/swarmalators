from threading import Thread
import time
import uvc
from typing import NamedTuple
import subprocess
import os
class CameraSpec(NamedTuple):
    uid: str
    width: int
    height: int
    fps: int
    bandwidth_factor: float = 2.0
class VideoStream:
    """A CV2 VideoStream wrapper for threading.

    Attributes:
        device: The device index of the camera to use.
    """

    def __init__(self, device: CameraSpec):
        self._stream = self._init_camera(device)

        # Apply camera settings
        input_config = open("default_camera.json", "r")

        is_windows = os.name == 'nt'
        res = subprocess.run(["uvcc", "--product", "2115", "import"], shell=is_windows, stdin=input_config, capture_output=True, text=True) 
        input_config.close()
        print(res.stdout)
        print(res.stderr)

        self._frame = self._stream.get_frame_robust()

        self._stopped = False

    def _init_camera(self, device: CameraSpec):
        cam = uvc.Capture(device.uid)

        cam.bandwidth_factor = device.bandwidth_factor

        for modes in cam.available_modes:
            if modes[:3] == (device.width, device.height, device.fps):
                cam.frame_mode = modes
                break
        else:
            cam.close()
            raise RuntimeError("Camera does not support the specified mode")
        
        return cam

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
                self._frame = self._stream.get_frame_robust()
            except:
                continue
    
    def read(self):
        """Return the current frame."""
        if self._frame.data_fully_received:
            data = self._frame.bgr if hasattr(self._frame, "bgr") else self._frame.gray
            return data
    
    def stop(self):
        """Indicate that the thread should be stopped."""
        self._stopped = True
        # Wait a moment to avoid segfaults
        time.sleep(0.5)
        # Release the stream
        self._stream.close()
        time.sleep(1)
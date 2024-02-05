import cv2
from threading import Thread
import time
import uvc
from typing import NamedTuple
import subprocess
class CameraControls(NamedTuple):
    brightness: int
    contrast: int
    saturation: int
    sharpness: int
    zoom: int
    gain: int
    auto_exposure_mode: int
    exposure_time: int
    auto_focus: int
    focus: int

camera_controls_mapping = {
    'brightness': 'Brightness',
    'contrast': 'Contrast',
    'saturation': 'Saturation',
    'sharpness': 'Sharpness',
    'zoom': 'Zoom absolute control',
    'gain': 'Gain',
    'auto_exposure_mode': 'Auto Exposure Mode',
    'exposure_time': 'Absolute Exposure Time',
    'auto_focus': 'Auto Focus',
    'focus': 'Absolute Focus'
}

DEFAULT_CAMERA_CONTROLS = CameraControls(
    brightness=150,
    contrast=128,
    saturation=255,
    sharpness=255,
    zoom=104,
    gain=150,
    auto_exposure_mode=1,
    exposure_time=500,
    auto_focus=0,
    focus=0
)
class CameraSpec(NamedTuple):
    uid: str
    width: int
    height: int
    fps: int
    bandwidth_factor: float = 2.0
    controls: CameraControls = DEFAULT_CAMERA_CONTROLS
class VideoStream:
    """A CV2 VideoStream wrapper for threading.

    Attributes:
        device: The device index of the camera to use.
    """

    def __init__(self, device: CameraSpec):
        self._stream = self._init_camera(device)

        # # Apply camera settings
        # print("Applying camera controls")
        # apply_camera_controls(controls)
        # print("Applied camera controls")

        # Apply camera settings
        input_config = open("default_camera.json", "r")
        res = subprocess.run(["uvcc", "--product", "2115", "import"], stdin=input_config, capture_output=True, text=True) 
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
        
        # controls_dict = dict([(c.display_name, c) for c in cam.controls])

        # for control, value in device.controls._asdict().items():
        #     uvc_name = camera_controls_mapping[control]

        #     controls_dict[uvc_name].value = value

        #     print("Set", uvc_name, "to", controls_dict[uvc_name].value)
        
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
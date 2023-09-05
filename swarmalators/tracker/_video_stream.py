import cv2
import uvc.uvc_bindings as uvc
from typing import NamedTuple, Optional
from threading import Thread


class CameraControls(NamedTuple):
    """
    Available camera controls.

    Attributes:
        brightness: Brightness
        contrast: Contrast
        saturation: Saturation
        sharpness: Sharpness
        zoom: Zoom absolute control
        gain: Gain
        exposure_mode: Auto Exposure Mode
        exposure_time: Absolute Exposure Time
    """

    brightness: int
    contrast: int
    saturation: int
    sharpness: int
    zoom: int
    gain: int
    exposure_mode: int
    exposure_time: int

    CONTROLS_NAME_MAP = {
        "brightness": "Brightness",
        "contrast": "Contrast",
        "saturation": "Saturation",
        "sharpness": "Sharpness",
        "zoom": "Zoom absolute control",
        "gain": "Gain",
        "exposure_mode": "Auto Exposure Mode",
        "exposure_time": "Absolute Exposure Time",
    }


class CameraSpec(NamedTuple):
    """
    Camera Specifications

    Attributes:
        name: Name
        width: Camera image width
        height: Camera image height
        fps: Camera FPS
        controls: Controls for camera
        bandwidth_factor: Scaling factor to control the bandwidth allocation over USB
    """

    name: str
    width: int
    height: int
    fps: int
    controls: CameraControls
    bandwidth_factor: float = 2.0


DEFAULT_CAMERA_SPEC = CameraSpec(
    name="Logitech Webcam C930e",
    width=1920,
    height=1080,
    fps=30,
    bandwidth_factor=1.6,
    controls=CameraControls(
        brightness=64,
        contrast=128,
        saturation=255,
        sharpness=255,
        zoom=100,
        gain=65,
        exposure_mode=1,
        exposure_time=312,
    ),
)


class VideoStream:
    """A CV2 VideoStream wrapper for threading.

    Attributes:
        device: The device index of the camera to use.
    """

    def __init__(self, device_index, camera_spec: CameraSpec = DEFAULT_CAMERA_SPEC):
        devices = uvc.device_list()

        device = devices[device_index]

        self._camera = self.init_camera(device, camera_spec)

        self._frame = None

        self._stopped = False

    def init_camera(self, device, camera: CameraSpec) -> Optional[uvc.Capture]:
        """
        Initialize a camera with the given specifications.

        Args:
            device: The device to initialize.
            camera: The camera specifications.

        Returns:
            The initialized camera.
        """
        capture = uvc.Capture(device["uid"])
        capture.bandwidth_factor = camera.bandwidth_factor

        for mode in capture.available_modes:
            if mode[:3] == camera[1:4]:  # compare width, height, fps
                capture.frame_mode = mode
                break
        else:
            print("None of the available modes match")
            capture.close()
            return

        controls_dict = dict([(c.display_name, c) for c in capture.controls])

        for key, value in camera.controls._asdict().items():
            try:
                controls_dict[CameraControls.CONTROLS_NAME_MAP[key]].value = value
            except KeyError:
                print(f"Unknown control: {key}")

        return capture

    def start(self):
        """Start the thread to read frames from the video stream."""
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        """Keep looping indefinitely until the thread is stopped."""
        try:
            while not self._stopped:
                try:
                    frame = self._camera.get_frame(timeout=0.001)
                except TimeoutError:
                    continue
                except uvc.InitError as err:
                    print(f"Failed to init: {err}")
                    self._stopped = True
                    break
                except uvc.StreamError as err:
                    print(f"Failed to get a frame: {err}")
                else:
                    data = frame.bgr if hasattr(frame, "bgr") else frame.gray
                    if frame.data_fully_received:
                        self._frame = data
        except KeyboardInterrupt:
            pass

    def read(self):
        """Return the current frame."""
        return self._frame

    def stop(self):
        """Indicate that the thread should be stopped."""
        self._stopped = True
        # Release the stream
        self._camera.close()

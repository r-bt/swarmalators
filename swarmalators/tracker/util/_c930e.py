from typing import NamedTuple
import subprocess
import os

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
        exposure_time: Absolute Exposure Time
    """

    brightness: int
    contrast: int
    saturation: int
    sharpness: int
    zoom: int
    gain: int
    # exposure_mode: int
    exposure_time: int
    auto_focus: int
    focus: int

    CONTROLS_NAME_MAP = {
        'exposure_time': 'exposure-time-abs',
        'zoom': 'zoom-abs',
        'auto_focus': 'auto-focus',
        'focus': 'focus-abs'
    }

PRODUCT = '0x046d:0x0843'

def apply_camera_controls(controls: CameraControls):
    """
    Applys UVC controls to the webcam

    Args:
        controls: The controls to apply
    """

    file_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(file_path)

    path_to_exc = os.path.join(dir_path, 'uvc-util')

    control_keys = controls._asdict().keys()

    manual_exposure = ('gain' in control_keys or 'exposure_time' in control_keys)

    subprocess.run([path_to_exc, "-V", PRODUCT, '-s', f"auto-exposure-mode={1 if manual_exposure else 8}"], stdout=subprocess.PIPE, text=True) 

    for control, value in controls._asdict().items():

        if control in CameraControls.CONTROLS_NAME_MAP:
            control = CameraControls.CONTROLS_NAME_MAP[control]
        
        subprocess.run([path_to_exc, "-V", PRODUCT, '-s', f"{control}={value}"], stdout=subprocess.PIPE, text=True) 
import uvc


def apply_settings(settings: dict[str, int], index=0):
    """Applies the given settings to the camera at the given index"""
    devices = uvc.device_list()
    device = devices[index]

    try:
        cap = uvc.Capture(device["uid"])
    except:
        print("Failed to open camera")
        return

    controls_dict = dict([(c.display_name, c) for c in cap.controls])

    print(controls_dict["Auto Focus"])

    for name, value in settings.items():
        controls_dict[name].value = value

    cap.close()

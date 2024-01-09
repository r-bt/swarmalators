import time
from tracker import Tracker, DirectionFinder
from nRFSwarmalator import nRFSwarmalator
import pdb
import numpy as np
import colorsys
import os
from swarmalator import Swarmalator

## NOTE: MODIFY TO THE PORT ON YOUR COMPUTER
PORT = "/dev/tty.usbmodem0010500746993"

SPHERO_SPEED_SCALE_FACTOR = 25

def init_spheros(swarmalator: nRFSwarmalator, finder: DirectionFinder):
    boxes = []

    swarmalator.set_mode(1)

    remaining_spheros = 14
    while remaining_spheros > 0: 
        swarmalator.matching_orientation()

        time.sleep(1.25) # Setting up the arrow animation takes time

        """
        Correct the orientation
        """

        while True:
            direction = finder.find_sphero_direction()

            if direction is None:
                continue

            heading = direction - 90

            print(heading)

            if heading < 0:
                heading += 360

            swarmalator.matching_correct_heading(heading)

            break

        """
        Get the bounding box to initalize the tracker
        """

        boxes.append(finder.find_sphero())

        """
        Increment sphero count
        """

        remaining_spheros -= 1

        if (remaining_spheros > 0):
            swarmalator.matching_next_sphero()
        
    swarmalator.set_mode(0)

    return boxes

def angles_to_rgb(angles_rad):
    # Convert the angles to hue values (ranges from 0.0 to 1.0 in the HSV color space)
    hues = angles_rad / (2 * np.pi)

    # Set fixed values for saturation and value (you can adjust these as desired)
    saturation = np.ones_like(hues)
    value = np.ones_like(hues)

    hsv_colors = np.stack((hues, saturation, value), axis=-1)
    rgb_colors = np.apply_along_axis(lambda x: colorsys.hsv_to_rgb(*x), -1, hsv_colors)

    # Scale RGB values to 0-255 range
    rgb_colors *= 255
    rgb_colors = rgb_colors.astype(np.uint8)

    return rgb_colors


def main():
    spheros = 15
    # Get the tracker
    direction_finder = DirectionFinder()

    # # Open connection to nRFSwarmalator
    nrf_swarmalator = nRFSwarmalator(PORT)

    boxes = init_spheros(nrf_swarmalator, direction_finder)

    # direction_finder.debug_show_boxes(boxes)

    # Release camera to transfer to tracking
    direction_finder.stop()

    # Wait for the camera to be released
    time.sleep(1)

    # Start tracking
    tracker = Tracker()

    tracker.start_tracking_objects(len(boxes), boxes)
    # tracker.start_tracking_objects(15, [])

    # Set the correct swarmalator mode
    nrf_swarmalator.set_mode(2)

    # """
    # Implements the Swarmalator model

    # Parameters
    # ----------
    # spheros : int
    #     Number of spheros to work with
    # K : int
    #     Phase coupling coefficient
    # """
    # spheros = 15
    # K = 1

    # """
    # Each agent has an inital angular frequency (column 1) and phase (column 2)
    # Each agent has an inital position (column 1 and 2) and velocity (column 3)

    # We set all angular frequency to 0

    # We randomize all inital phases between [0, 2 * pi]
    # """

    # Get the positions from the tracker
    positions = tracker.get_positions()
    got = False
    while not got:
        try:
            positions = tracker.get_positions()
            if positions is not None:
                got = True
        except:
            continue

    # Init the swarmalator model
    swarmalator = Swarmalator(spheros, 0, 1)

    swarmalator.update(positions[:, :2])

    count = 0

    while True:
        try:
            positions = tracker.get_positions()

            if positions is None:
                count += 1
                if (count == 100):
                    print("100 tries without position")
                continue

            count = 0

            swarmalator.update(positions[:, :2])

            phase_state = swarmalator.get_phase_state()

            velocity = swarmalator.get_velocity()

            velocities = []
            for v in velocity:
                speed = int(np.linalg.norm(v) * SPHERO_SPEED_SCALE_FACTOR)
                heading = int(np.degrees(np.arctan2(v[1], v[0])))
                if heading < 0:
                    heading += 360
                velocities.append((speed, heading))

            print(velocities)

            colors = angles_to_rgb(phase_state[:, 1])

            nrf_swarmalator.colors_set_colors(colors, velocities)
        except Exception as e:
            print(e)
            # continue

if __name__ == "__main__":
    if os.name == "posix" and os.geteuid() != 0:
        print("This script must be run as root!")
        exit(1)

    main()

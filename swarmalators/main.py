import time
from tracker import Tracker, DirectionFinder
from nRFSwarmalator import nRFSwarmalator
import pdb
import numpy as np
import colorsys
import os
from swarmalator import Swarmalator
import cv2
from simple_pid import PID
import csv
import os

## NOTE: MODIFY TO THE PORTs ON YOUR COMPUTER FOR THE NRF5340
is_windows = os.name == "nt"
PORT1 = "/dev/tty.usbmodem0010500530493" if not is_windows else "COM7"
PORT2 = "/dev/tty.usbmodem0010500746993" if not is_windows else "COM9"

MAX_SPHEROS_PER_SWARMALATOR = 15

CAMERA_INDEX = 0


def init_spheros(
    spheros: int, swarmalators: list[nRFSwarmalator], finder: DirectionFinder
):
    boxes = []

    for swarmalator in swarmalators:
        swarmalator.set_mode(1)

    remaining_spheros = spheros
    while remaining_spheros > 0:
        swarmalator_index = (spheros - remaining_spheros) // MAX_SPHEROS_PER_SWARMALATOR
        swarmalator = swarmalators[swarmalator_index]

        swarmalator.matching_orientation()

        time.sleep(1.25)  # Setting up the arrow animation takes time

        """
        Correct the orientation

        Notes:
        1. Sphero Logo (not text) is the front
        2. 90 degrees is the right side
        3. 180 degrees is the back
        4. 270 degrees is the left side
        """

        while True:
            direction = finder.find_sphero_direction()

            if direction is None:
                continue

            heading = direction

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

        if remaining_spheros > 0:
            swarmalator.matching_next_sphero()

        print("Sphero calibrated! Remaining spheros: {}".format(remaining_spheros))

    for swarmalator in swarmalators:
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
    spheros_old = [
        "SB-1B35",
        "SB-F860",
        "SB-2175",
        "SB-369C",
        "SB-618E",
        "SB-6B58",
        "SB-9938",
        "SB-BFD4",
        "SB-C1D2",
        "SB-CEFA",
        "SB-DF1D",
        "SB-F465",
        "SB-F479",
        "SB-F885",
        "SB-FCB2",
    ]

    spheros_new = [
        "SB-31B8",
        "SB-9CA8",
        "SB-80C4",
        "SB-F509",
        "SB-5883",
        "SB-8893",
        "SB-D64E",
        "SB-7D72",
        "SB-7D7C",
        "SB-4483",
        "SB-378F",
        "SB-2C58",
        "SB-D9E2",
        "SB-2E4B",
        "SB-6320",
    ]

    # # We have to figure out which camera is for tracking and which for recording
    # camera_index = 0

    # while camera_index < 2:
    #     cam = cv2.VideoCapture(camera_index)
    #     user_input = input("Is camera {} the tracking camera? (y/n): ".format(camera_index))
    #     cam.release()
    #     if user_input == "y":
    #         break
    #     else:
    #         camera_index += 1
    #         if camera_index == 2:
    #             print("No tracking camera found!")
    #             exit(1)

    # Start the direction finder
    direction_finder = DirectionFinder(CAMERA_INDEX)

    # Start connects to Nordic boards

    nrf_swarmalator_1 = nRFSwarmalator(spheros_new, PORT1)
    nrf_swarmalator_2 = nRFSwarmalator(spheros_old, PORT2)

    # Specify the NORDIC boards to be used
    swarmalators = [nrf_swarmalator_1, nrf_swarmalator_2]

    for swarmalator in swarmalators:
        swarmalator.wait_for_spheros()

    # Calibrate all the spheros
    spheros = [*spheros_old, *spheros_new]

    boxes = init_spheros(len(spheros), swarmalators, direction_finder)

    # Release camera to transfer to tracking
    direction_finder.stop()

    # Wait for the camera to be released
    time.sleep(1)

    # Set the correct swarmalator mode
    for nrf_swarmalator in swarmalators:
        nrf_swarmalator.set_mode(2)

    # Start tracking
    tracker = Tracker()

    print("Init tracker")

    tracker.start_tracking_objects(CAMERA_INDEX, len(boxes), boxes)

    print("Started tracking")

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

    print(positions)

    # Init the swarmalator model
    swarmalator = Swarmalator(len(spheros), 0.5, 1)
    swarmalator.update(positions[:, :2])

    # count = 0

    now = time.monotonic()

    prev_positions = None

    # Set up PID controller to control speeds
    Kp = 50
    Ki = 1
    Kd = 0

    pid_controllers = [PID(Kp, Ki, Kd, setpoint=0) for _ in range(len(spheros))]
    for pid in pid_controllers:
        pid.output_limits = (0, 100)
        pid.sample_time = 0.1

    current_time = time.strftime("%Y%m%d%H%M%S")
    output_file = f"state_{current_time}.csv"

    with open(output_file, "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(
            [
                "Time",
                *["Phase {}".format(i) for i in range(len(spheros))],
                *["Position {}".format(i) for i in range(len(spheros))],
            ]
        )

        while True:
            try:
                # Update and get values from swarmalator model
                positions = tracker.get_positions()

                if positions is None:
                    continue

                swarmalator.update(positions[:, :2])
                swarmalator.update_phase(time.monotonic() - now)

                phase_state = swarmalator.get_phase_state()
                velocities = swarmalator.get_velocity()

                # Calculate the current velocity for all Spheros
                real_velocities = np.zeros(len(spheros))
                if prev_positions is not None:
                    traveled = np.linalg.norm(
                        positions[:, :2] - prev_positions[:, :2], axis=1
                    )
                    real_velocities = traveled / (time.monotonic() - now)

                prev_positions = positions
                now = time.monotonic()

                # Update the PID controllers to get new velocities

                to_send_velocities = []
                for i, (controller, velocity) in enumerate(
                    zip(pid_controllers, velocities)
                ):
                    # Update the set point to the new desired velocity
                    controller.setpoint = np.linalg.norm(velocity)

                    # Update the PID controller
                    command = controller(real_velocities[i])

                    # Get the heading
                    heading = int(np.degrees(np.arctan2(velocity[1], velocity[0])))
                    # Sphero uses a different heading system (0 is the front, 90 is the right side, 180 is the back, 270 is the left side)
                    # Effect is that left and right are switched
                    heading = -heading

                    if heading < 0:
                        heading += 360

                    # Store the speed and heading
                    to_send_velocities.append((int(command), heading))

                colors = angles_to_rgb(phase_state[:, 1])

                for i, nrf_swarmalator in enumerate(swarmalators):
                    color_selection = colors[
                        i
                        * MAX_SPHEROS_PER_SWARMALATOR : (i + 1)
                        * MAX_SPHEROS_PER_SWARMALATOR
                    ]

                    velocities_selection = to_send_velocities[
                        i
                        * MAX_SPHEROS_PER_SWARMALATOR : (i + 1)
                        * MAX_SPHEROS_PER_SWARMALATOR
                    ]

                    nrf_swarmalator.colors_set_colors(
                        color_selection, velocities_selection
                    )

                tracker.set_velocities([(v[0], -v[1]) for v in to_send_velocities])

                print("Took: ", time.monotonic() - now)

                writer.writerow(
                    [time.monotonic(), *phase_state[:, 1], *positions[:, :2]]
                )
            except Exception as e:
                print(e)
                # continue


if __name__ == "__main__":
    if os.name == "posix" and os.geteuid() != 0:
        print("This script must be run as root!")
        exit(1)

    main()

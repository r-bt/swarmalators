import time
from tracker import Tracker, DirectionFinder
from nRFSwarmalator import nRFSwarmalator
import pdb
import numpy as np
import colorsys
import os

## NOTE: MODIFY TO THE PORT ON YOUR COMPUTER
PORT = "/dev/tty.usbmodem0010500746993"


def init_spheros(swarmalator: nRFSwarmalator, finder: DirectionFinder):
    boxes = []

    swarmalator.set_mode(1)

    remaining_spheros = 15
    while remaining_spheros > 0: 
        swarmalator.matching_orientation()

        time.sleep(0.75) # Setting up the arrow animation takes time

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


# def non_blocking_input(prompt=''):
#     sys.stdout.write(prompt)
#     sys.stdout.flush()
#     rlist, _, _ = select.select([sys.stdin], [], [], 0)
#     if rlist:
#         return sys.stdin.readline().rstrip()
#     return None


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
    # Get the tracker
    direction_finder = DirectionFinder()

    # # Open connection to nRFSwarmalator
    swarmalator = nRFSwarmalator(PORT)

    boxes = init_spheros(swarmalator, direction_finder)

    # swarmalator.set_mode(2)

    # angle = 0
    # while True:
    #     angle += 1
    #     angle %= 360

    #     rgb = colorsys.hsv_to_rgb(angle / 360, 1, 1)

    #     rgb = [int(x * 255) for x in rgb]

    #     swarmalator.colors_set_colors([rgb for _ in range(15)])

    # swarmalator.colors_set_colors()

    # swarmalator.mode = 1

    # while True:
        # swarmalator.set_mode(1)

    # swarmalator.matching_orientation()

    # swarmalator.matching_orientation()
# 
        # swarmalator.matching_correct_heading(0)

        # time.sleep(1)

    # # Get inital boxes and directions of spheros
    # boxes = init_spheros(swarmalator, direction_finder)

    direction_finder.debug_show_boxes(boxes)

    # Release camera and transfer to tracking
    direction_finder.stop()

    # tracker = Tracker()

    # tracker.start_tracking_objects(len(boxes), boxes)

    # while True:
    #     tracker.get_positions()
    
    # ser.write(bytearray([0x8d, 0x01, 0x01, 0x0a]))

    # time.sleep(1)

    # ser.write(bytearray([0x8d, 0x02, 0x0a]))
    
    # ser.write(bytearray([0x01]))

    # while True:
    #     print("Sending...")
    #     ser.write(bytearray([0x8d, 0x01, 0x0a]))
    #     # ser.write(bytearray([0x8d, 0x02, 0x0a]))

    #     time.sleep(1)

        # ser.write(bytearray([0x8d, 0x03, 0x0a]))

        # time.sleep(5)

    # box = direction_finder.find_sphero()
    # while True:
    #     direction = direction_finder.find_sphero_direction()

    #     if direction is None:
    #         continue

    #     heading = direction - 90

    #     if heading < 0:
    #         heading += 360

    #     # Split the angle into two bytes
    #     byte1 = heading // 256  # Most significant byte
    #     byte2 = heading % 256  # Least significant byte

    #     print(direction)
    #     print(heading)

    #     data = bytearray([byte1, byte2]) + b"\n"

    #     print(data)

    #     ser.write(data)

    #     time.sleep(0.2)

    #     break

    # while True:
    #     direction = direction_finder.find_sphero_direction()

    #     if direction is None:
    #         continue

    #     print(direction)
    # tracker = Tracker()

    # tracker.get_single_sphero_direction()

    # ser = serial.Serial(PORT, 115200, timeout=5)  # open serial port

    # ser.close()
    # ser.open()
    # if ser.isOpen():
    #     print(ser.portstr, ":connection successful.")
    # else:
    #     print("Error opening serial port")
    #     return

    # # ser.write(bytearray([2]))

    # angle = 0

    # while True:
    #     angle += 1
    #     angle %= 360

    #     rgb = colorsys.hsv_to_rgb(angle / 360, 1, 1)

    #     rgb = [int(x * 255) for x in rgb]

    #     rgbs = [item for _ in range(15) for item in rgb]

    #     # print(bytearray(rgbs))

    #     ser.write(bytearray(rgbs))

    #     # time.sleep(0.5)

    #     try:
    #         start = time.time()
    #         data = ser.readline()
    #         end = time.time()
    #         print("Took: ", end - start)
    #     except serial.SerialException:
    #         print("Caught exception!")
    #         ser.close()
    #         ser = serial.Serial(PORT, 115200, timeout=5)
    #         data = ser.readline()

    #     if len(data) < 1:
    #         print("No data recevied!")
    #         break

    #     if data[0] != 0x8D:
    #         print("Invalid packet")
    #         print(data)
    #         break

    # try:
    #     data = ser.readline()
    # except serial.SerialException:
    #     print("Caught exception!")
    #     ser.close()
    #     ser = serial.Serial(PORT, 115200, timeout=5)
    #     data = ser.readline()

    # if len(data) < 1:
    #     print("No data recevied!")
    #     break

    # if data[0] != 0x8D:
    #     print("Invalid packet")
    #     print(data)
    #     break

    # x = data[1]

    # print("Received: ", x)

    # colors = [item for _ in range(15) for item in (255, 0, 0)]

    # print(bytearray(colors))

    # ser.write(bytearray(colors))

    # init_positions = init_spheros(ser, tracker)

    # print(init_positions)

    # spheros = 15

    # tracker.start_tracking_objects(spheros, init_positions)

    # # Activate color state
    # ser.write(bytearray([2]))

    # """
    # Implements the Swarmalator model

    # Parameters
    # ----------
    # spheros : int
    #     Number of spheros to work with
    # K : int
    #     Phase coupling coefficient
    # """
    # K = 1

    # """
    # Each agent has an inital angular frequency (column 1) and phase (column 2)

    # We set all angular frequency to 0

    # We randomize all inital phases between [0, 2 * pi]
    # """

    # state = np.random.rand(spheros, 2)
    # # state[:, 0] = np.pi
    # # state[:, 0] = 0
    # state[:, 1] *= 2 * np.pi

    # now = time.time()

    # while True:
    #     try:
    #         positions = tracker.get_positions()
    #         if positions is None:
    #             continue

    #         # First we will calculate the difference of the phases
    #         phases = state[:, 1:]

    #         phase_sin_difference = np.sin(phases.T - phases)

    #         # Now we will calculate the unit vectors
    #         vectors = positions[:, :2][:, np.newaxis] - positions[:, :2]
    #         pairwise_distances = np.linalg.norm(vectors, axis=2)

    #         mask = (pairwise_distances != 0)

    #         sums = np.sum(np.where(mask, phase_sin_difference / pairwise_distances, 0), axis=1)

    #         # Calculate the new state
    #         delta_phases = state[:, 0] + (K/spheros) * sums
    #         state[:, 1] += delta_phases * (time.time() - now)

    #         # Bound the values between 0 and 2 pi
    #         state[:, 1] %= 2 * np.pi

    #         colors = angles_to_rgb(state[:, 1])

    #         tx = bytearray(colors.flatten())

    #         # time.sleep(0.2)

    #         print(tx)

    #         ser.write(tx)
    #     except:
    #         continue


if __name__ == "__main__":
    if os.name == "posix" and os.geteuid() != 0:
        print("This script must be run as root!")
        exit(1)

    main()

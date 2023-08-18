import serial
import time
from tracker import Tracker
import pdb
import atexit

## NOTE: MODIFY TO THE PORT ON YOUR COMPUTER
PORT = '/dev/tty.usbmodem0010502148741'

def init_spheros(tracker):
    boxes = []
    ser = serial.Serial(PORT, 115200, timeout=5)  # open serial port

    ser.close()
    ser.open()
    if ser.isOpen():
        print(ser.portstr, ":connection successful.")
    else:
        print("Error opening serial port")
        return

    x = 3
    while x > 0:
        ser.write(bytearray([1]))

        try:
            data = ser.readline()
        except serial.SerialException:
            print("Caught exception!")
            ser.close()
            ser = serial.Serial(PORT, 115200, timeout=5)
            data = ser.readline()

        if len(data) < 1:
            print("No data recevied!")
            break

        if data[0] != 0x8d:
            print("Invalid packet")
            print(data)
            break

        x = data[1]

        print("Remaining: ", x)

        time.sleep(0.2)

        box = tracker.find_single_sphero()

        boxes.append(box)

        ser.reset_input_buffer()
    
    ser.write(bytearray([0]))

    ser.close()

    return boxes

def main():
    tracker = Tracker(15)

    atexit.register(tracker.cleanup)

    init_positions = init_spheros(tracker)

    tracker.start_tracking_objects(init_positions)

    while True:
        cmd = input()
        if cmd == 'q':
            print("cleaning up...")
            tracker.cleanup()
            return

if __name__ == '__main__':
    main()
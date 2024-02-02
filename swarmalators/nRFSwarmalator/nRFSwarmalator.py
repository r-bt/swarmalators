import serial
import time

class nRFSwarmalator():

    def __init__(self, spheros: int, port):
        self.port = port
        self.spheros = spheros

        self.ser = serial.Serial(self.port, 115200, timeout=5, rtscts=False)  # open serial port

        self.ser.close()
        self.ser.open()

        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

        if not self.ser.isOpen():
            print("Error opening serial port")
            return
        
        self.mode = 0
        
        # Reset to initalize state
        self.reset()

    def reset(self):
        """
        Resets the state on the nRFSwarmalator
        """
        self._send_command(bytearray([0x00]))

    def set_mode(self, mode: int):
        """
        Sets the mode of the nRFSwarmalator

        Args:
            mode (int): The mode to set the nRFSwarmalator to
        """
        self.reset()
        self._send_command(bytearray([0x01, mode]))

        self.mode = mode

    def matching_next_sphero(self):
        if (self.mode != 1):
            raise RuntimeError("Mode must be MATCHING to use this function")
        self._send_command(bytearray([0x01]))

    def matching_fill_matrix(self):
        if (self.mode != 1):
            raise RuntimeError("Mode must be MATCHING to use this function")
        self._send_command(bytearray([0x02]))

    def matching_orientation(self):
        if (self.mode != 1):
            raise RuntimeError("Mode must be MATCHING to use this function")
        self._send_command(bytearray([0x03]))

    def matching_correct_heading(self, heading):
        """
        Corrects the heading of the sphero by turning the sphero and then resetting its aim

        Args:
            heading (int): The heading to correct to
        """
        if (self.mode != 1):
            raise RuntimeError("Mode must be MATCHING to use this function")

        # Split the angle into two bytes
        byte1 = heading // 256  # Most significant byte
        byte2 = heading % 256  # Least significant byte

        self._send_command(bytearray([0x04, byte1, byte2]))

    def colors_set_colors(self, colors: list[int], velocities: list[int]):
        if (self.mode != 2):
            raise RuntimeError("Mode must be COLORS to use this function")

        if (len(colors) != self.spheros):
            raise RuntimeError("Colors must be a list of {} RGB values".format(self.spheros))
        
        if (len(velocities) != self.spheros):
            raise RuntimeError("Must be a list of {} velocities".format(self.spheros))
        
        rgbs = [x for item in colors for x in item]
        velocities = [(speed, heading // 256, heading % 256) for (speed, heading) in velocities]
        rgbs.extend([x for item in velocities for x in item])

        self._send_command(bytearray([0x01, *rgbs]))

    """
    PRIVATE
    """

    def _receive_response(self) -> bytearray:
        """
        Waits for nRFSwarmalator to send data. Then verifies the packet valid and returns the data

        Returns:
            bytearray: The data received from the nRFSwarmalator
        """
        try:
            data = self.ser.readline()
        except serial.SerialException:
            print("Exception!")
            return None

        if len(data) < 1:
            print("No data received!")
            return None
        
        if data[0] != 0x8D:
            print("Invalid packet")
            print(data)
            return None
        
        return data[1:-1]

        
    def _send_command(self, data: bytearray):
        """
        Send command to the nRFSwarmalator

        Args:
            data (bytearray): The data to send to the nRFSwarmalator

        Returns:
            bytearray: The data received from the nRFSwarmalator
        """
        self.ser.reset_input_buffer()

        res = self.ser.write(bytearray([0x8d, *data, 0x0a]))

        data = self._receive_response()

        if data is None:
            print("Error sending command!")
            exit()
        else:
            return data

    
    


    
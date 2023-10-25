import cv2
from ._video_stream import VideoStream, CameraControls, CameraSpec
from .util._c930e import apply_settings
import numpy as np

CONTOUR_DISTANCE_THRESHOLD = 350

class DirectionFinder:
    """
    DirectionFinder. Identify the direction of a single sphero
    """

    MAX_HISTORY_LEN = 40

    CAMERA_SPEC = CameraSpec(
        name="Logitech Webcam C930e",
        width=1920,
        height=1080,
        fps=30,
        bandwidth_factor=1.6,
        controls=CameraControls(
            brightness=100,
            contrast=128,
            saturation=255,
            sharpness=255,
            zoom=102,
            gain=64,
            exposure_mode=1,
            exposure_time=312,
        ),
    )

    def __init__(self):
        # Apply settings
        # apply_settings(self.CAMERA_SETTINGS)
        # Get camera on OpenCV
        self.stream = VideoStream(0, self.CAMERA_SPEC).start()
        # Store history of frames for direction finding
        self.history = []

    def __del__(self):
        self.stream.stop()

    def stop(self):
        self.stream.stop()

    def find_sphero_direction(self):
        """
        Find the direction of a single sphero

        We find the direction from the x-axis counter clockwise
        """
        frame = self._get_frame()

        if frame is None:
            return None

        # Find the largest contour
        contours = self._find_all_contours(frame)

        if len(contours) == 0:
            return None

        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        # Get largest contour (this is the base)
        contour_base = contours[0]

        # Get the center of the base contour
        M_base = cv2.moments(contour_base)
        cX_base = M_base["m10"] / M_base["m00"]
        cY_base = M_base["m01"] / M_base["m00"]

        # Find the largest contour within a certain distance (this is the head)

        for contour in contours[1:]:
            # Get the center of the head contour
            M_head = cv2.moments(contour)
            cX_head = M_head["m10"] / M_head["m00"]
            cY_head = M_head["m01"] / M_head["m00"]
            # Check distance between this and the base contour
            if (cX_head - cX_base)**2 + (cY_head - cY_base)**2 < CONTOUR_DISTANCE_THRESHOLD:
                break
        else:
            print("Failed to find head contour")
            
            while True:
                frame2 = self.stream.read()
                cv2.imshow("Frame", frame2)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            return None

        # Calculate the direction vector from the base to the head
        direction_vector = np.array([cX_head - cX_base, cY_base - cY_head])

        angle_radians = np.arctan2(direction_vector[1], direction_vector[0])

        angle_degrees = np.degrees(angle_radians)

        if angle_degrees < 0:
            angle_degrees += 360

        # image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # text = f"Angle: {angle_degrees:.2f} degrees"
        # cv2.putText(
        #     image,
        #     text,
        #     (int(cX_head) + 10, int(cY_head) + 10),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (255, 0, 255),
        #     2,
        # )

        # cv2.line(
        #     image,
        #     (int(cX_base), int(cY_base)),
        #     (int(cX_head), int(cY_head)),
        #     (255, 0, 0),
        #     2,
        # )

        # while True:
        #     frame2 = self.stream.read()
        #     cv2.imshow("Lines and Intersection", image)
        #     cv2.imshow("Frame", frame2)

        #     if cv2.waitKey(1) & 0xFF == ord("q"):
        #         break

        return int(angle_degrees)

    def find_sphero(self):
        """
        Find the location of a single (lit) sphero

        Assumes:
            Only one Sphero's LED matrix is on

        Returns:
            The bounding box of the sphero
        """
        # Attempt this 10 times
        box = None

        for _ in range(0, 75):
            frame = self.stream.read()

            if frame is None:
                continue

            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed_frame = cv2.GaussianBlur(processed_frame, (21, 21), 0)
            processed_frame = cv2.threshold(
                processed_frame, 150, 255, cv2.THRESH_BINARY
            )[1]
            processed_frame = cv2.erode(processed_frame, None, iterations=1)
            processed_frame = cv2.dilate(processed_frame, None, iterations=12)

            contours = self._find_all_contours(processed_frame)

            if len(contours) == 0:
                continue

            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            contour = contours[0]

            x, y, w, h = cv2.boundingRect(contour)
            box = [x, y, x + w, y + h]

            break

        if box == None:
            print("Failed to find color")

        return box
    
    def debug_show_boxes(self, boxes):

        while True:

            frame = self.stream.read()

            for box in boxes:
                if box is None:
                    continue

                x, y, x_2, y_2 = box

                w = x_2 - x
                h = y_2 - y

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    """
    Private member functions
    """

    def _get_frame(self):
        """
        Get a frame from the video stream

        Processes frame and smooths with history of previous frames

        Returns:
            processed_frame (numpy.ndarray): Processed frame
        """
        frame = self.stream.read()

        if frame is None:
            print("Frame is None!")
            return None

        self.history.append(frame)

        if len(self.history) > self.MAX_HISTORY_LEN:
            self.history.pop(0)

        smoothed_frame = frame.copy()

        alpha = 1.0 / len(self.history)
        for hist_frame in self.history:
            cv2.addWeighted(
                hist_frame, alpha, smoothed_frame, 1 - alpha, 0, smoothed_frame
            )

        processed_frame = cv2.cvtColor(smoothed_frame, cv2.COLOR_BGR2GRAY)
        processed_frame = cv2.threshold(processed_frame, 70, 255, cv2.THRESH_BINARY)[1]

        processed_frame = cv2.erode(processed_frame, None, iterations=1)
        processed_frame = cv2.dilate(processed_frame, None, iterations=1)

        return processed_frame

    def _approx_contour_as_poly(self, contour):
        """
        Approximates a contour as a polygon

        Args:
            contour (numpy.ndarray): Contour to approximate

        Returns:
            approx (numpy.ndarray): Approximated polygon
        """
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        return approx

    def _get_longest_line(self, approx):
        """
        Finds the longest line in a contour

        Args:
            contour (numpy.ndarray): Contour to find longest line in

        Returns:
            max_line (tuple): Longest line in contour
        """

        max_line = None
        max_length = 0

        for i in range(len(approx)):
            start = tuple(approx[i][0])
            end = tuple(approx[(i + 1) % len(approx)][0])
            length = np.linalg.norm(np.array(start) - np.array(end))

            if length > max_length:
                max_length = length
                max_line = (start, end)

        return max_line

    def _get_longest_perpendicular_line(self, approx, line):
        """
        Finds the longest perpendicular line from the longest line to a point in the approximation

        Args:
            approx (numpy.ndarray): Contour to find point in
            line (tuple): Line to find point from

        Returns:
            max_perpendicular_line (numpy.ndarray): Longest perpendicular line
        """

        (x1, y1), (x2, y2) = line

        if (x2 - x1) == 0:
            return None

        m = (y2 - y1) / (x2 - x1)  # Slope
        c = y1 - m * x1  # Intercept

        max_perpendicular_distance = 0
        max_perpendicular_point = None

        for point in approx:
            x, y = point[0]
            perpendicular_distance = abs(((m * x + c) - y) / np.sqrt(m**2 + 1))

            if perpendicular_distance > max_perpendicular_distance:
                max_perpendicular_distance = perpendicular_distance
                max_perpendicular_point = (x, y)

        if max_perpendicular_point is None:
            return

        if m != 0:
            perpendicular_m = -1 / m
        else:
            perpendicular_m = float("inf")  # Vertical line

        perpendicular_c = (
            max_perpendicular_point[1] - perpendicular_m * max_perpendicular_point[0]
        )

        # Find the intersection point between max_line and the perpendicular line
        intersection_x = (perpendicular_c - c) / (m - perpendicular_m)
        intersection_y = m * intersection_x + c

        # Calculate the direction vector from the intersection point to max_perpendicular_point
        longest_perpendicular_line = np.array(
            [
                max_perpendicular_point[0] - intersection_x,
                intersection_y - max_perpendicular_point[1],
            ]
        )

        return longest_perpendicular_line

    """
    STATIC
    """

    @staticmethod
    def _find_all_contours(img):
        """
        Find all contours in an image

        Args:
            img: The image to find contours in

        Returns:
            A list of contours
        """
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

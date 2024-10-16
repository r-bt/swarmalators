import cv2
from ._video_stream import VideoStream
import multiprocessing as mp

# from ._sort import Sort
from .euclid_tracker import EuclideanDistTracker
import numpy as np
import time
import atexit

MAX_LEN = 1


def find_all_contours(img):
    """
    Find all contours in an image

    Args:
        img: The image to find contours in

    Returns:
        A list of contours
    """
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


class SpheroTracker:
    """
    Tracker Class. Handles actual tracking of Spheros

    Attributes:
        init_positions: The initial positions of the Spheros
    """

    def __init__(
        self,
        device: int,
        spheros: int,
        tracking: mp.Event,
        positions,
        lock,
        velocities,
        init_positions: list = [],
    ):
        self._spheros = spheros
        self._tracking = tracking
        self._positions = positions
        self._lock = lock
        self._velocities = velocities

        self._stream = VideoStream(device, "default_camera.json").start()

        # Get scale factors
        self._set_scale_factor()

        # It takes some time for the camera to focus, etc
        print("Waiting for camera to calibrate")
        self._calibrate_camera()
        print("Calibrated camera")

        # Setup and initalize tracker
        self._euclid_tracker = EuclideanDistTracker()
        self._initalize_tracker(init_positions)

        self._init_recording()

    """
    Private methods
    """

    def _init_recording(self):
        """
        Initalize recording
        """
        # Set up video writer using FFmpeg
        current_time = time.strftime("%Y%m%d%H%M%S")
        output_file = f"output_{current_time}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Change the codec as needed
        out = cv2.VideoWriter(
            output_file, fourcc, 20.0, (self._width, self._height)
        )  # Adjust parameters as needed
        self.out = out

        # Setup cleanup function
        def cleanup():
            out.release()

        atexit.register(cleanup)

    def _set_scale_factor(self):
        """
        Sets the scale factor so we can normalize coordinates
        """
        frame = self._stream.read()
        while frame is None:
            frame = self._stream.read()

        self._height, self._width = frame.shape[:2]

        self._scale_factor = 8.0 / max(self._width, self._height)

    def _calibrate_camera(self):
        count = 0
        while count < 100:
            frame = self._stream.read()

            if frame is None:
                continue

            thresh = self._process_frame(frame)

            dets = self._detect_objects(thresh)

            if len(dets) == self._spheros:
                count += 1
            else:
                print("Only found {} spheros".format(len(dets)))

    def _initalize_tracker(self, init_positions):
        if len(init_positions) > 0:
            self._euclid_tracker.init(init_positions)
        else:
            while True:
                frame = self._stream.read()

                if frame is None:
                    continue

                thresh = self._process_frame(frame)

                dets = self._detect_objects(thresh)

                if len(dets) == self._spheros:
                    self._euclid_tracker.init(dets)
                    break

    def _track_objects(self):
        """
        Track objects
        """

        frame_time = 0
        prev_frame_time = 0

        while self._tracking.is_set():
            frame = self._stream.read()
            self.out.write(frame)

            thresh = self._process_frame(frame)

            dets = self._detect_objects(thresh)

            # active_tracks = self._sort_tracker.update(dets)
            active_tracks = []
            try:
                active_tracks = self._euclid_tracker.update(dets)
            except RuntimeError as e:
                print("Error updating tracker")
                self._stream.stop()  # Stop the video stream
                self.out.release()
                print(e)
                while True:
                    cv2.imshow("Frame", frame)
                    cv2.imshow("Thresh", thresh)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise RuntimeError("User quit")

            pos = self._update_positions(active_tracks)

            # sorted_indices = np.argsort(pos[:, 2])

            with self._lock:
                if len(self._positions) >= MAX_LEN:
                    self._positions.pop(0)
                self._positions.append(pos)

            for index, (obj_id, track) in enumerate(active_tracks):
                if index < len(self._velocities):
                    velocity = self._velocities[index]

                    arrow_length = 40  # Adjust the arrow length as needed

                    heading_radians = np.radians(velocity[1])

                    # Calculate the endpoint of the arrow based on velocity and arrow_length
                    arrow_end = (
                        int(track[0] + arrow_length * np.cos(heading_radians)),
                        int(track[1] + arrow_length * np.sin(-heading_radians)),
                    )

                    # Draw the arrow on the frame
                    cv2.arrowedLine(
                        frame,
                        (int(track[0]), int(track[1])),
                        arrow_end,
                        (36, 255, 12),
                        5,
                    )

                    cv2.putText(
                        frame,
                        "Speed: {}, Heading: {}".format(
                            round(velocity[0], 2), round(velocity[1], 2)
                        ),
                        (int(track[0]), int(track[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (36, 255, 12),
                        2,
                    )

            # We can't handle losing Spheros yet (fix in the future)

            if len(active_tracks) < self._spheros:
                print("Not enough spheros detected")
                print([len(active_tracks), len(dets)])
                while True:
                    cv2.imshow("Frame", frame)
                    cv2.imshow("Thresh", thresh)

                    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit the loop
                        break

            # Calculate the fps
            frame_time = time.time()

            fps = 1 / (frame_time - prev_frame_time)
            prev_frame_time = frame_time

            fps = str(int(fps))

            cv2.putText(
                frame,
                fps,
                (7, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (100, 255, 0),
                3,
                cv2.LINE_AA,
            )

            # Display the image

            cv2.imshow("Frame", frame)
            cv2.imshow("Thresh", thresh)

            if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit the loop
                break

        print("Saving video...")
        cv2.destroyAllWindows()  # Close all OpenCV windows
        self._stream.stop()  # Stop the video stream
        self.out.release()

    def _detect_objects(self, frame):
        """
        Detect Spheros in a frame

        Args:
            frame: The frame to detect Spheros in

        Returns:
            A list of detections in form [x, y, w, h]
        """
        contours = find_all_contours(frame)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        endIndex = min(self._spheros, len(contours))

        dets = []
        for contour in contours[:endIndex]:
            x, y, w, h = cv2.boundingRect(contour)
            det = np.array([x, y, w, h])
            dets.append(det)

        return np.array(dets)

    def _update_positions(self, tracks):
        """
        Update the positions of the spheros

        Args:
            dets: The detections to update the positions with in form [id, (x, y)]
        """

        pos = np.empty((len(tracks), 3))

        for i, track in enumerate(tracks):
            center_x, center_y = self._normalize_coordinates(track[1][0], track[1][1])

            pos[i] = np.array([center_x, center_y, track[0]])

        return pos

    def _normalize_coordinates(self, x, y):
        normalized_x = self._scale_factor * (x - self._width / 2)
        normalized_y = self._scale_factor * (y - self._height / 2)
        return normalized_x, -normalized_y

    def _unnormalize_coordinates(self, x, y):
        unnormalized_x = (x / self._scale_factor) + self._width / 2
        unnormalized_y = (-y / self._scale_factor) + self._height / 2
        return unnormalized_x, unnormalized_y

    """
    Static methods
    """

    @staticmethod
    def start_tracking(
        device: int,
        spheros: int,
        tracking: mp.Event,
        positions,
        lock,
        velocities,
        init_positions: list = [],
    ):
        """
        Start tracking spheros
        """
        print("Starting to track!")
        sphero_tracker = SpheroTracker(
            device, spheros, tracking, positions, lock, velocities, init_positions
        )

        sphero_tracker._track_objects()

    @staticmethod
    def _process_frame(frame):
        """
        Process frame

        Args:
            frame: The frame to process

        Returns:
            The processed frame
        """
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_frame = cv2.GaussianBlur(processed_frame, (21, 21), 0)

        # processed_frame = cv2.adaptiveThr eshold(processed_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        processed_frame = cv2.threshold(processed_frame, 60, 255, cv2.THRESH_BINARY)[1]

        processed_frame = cv2.erode(processed_frame, None, iterations=4)
        processed_frame = cv2.dilate(processed_frame, None, iterations=2)

        return processed_frame


class Tracker:
    """
    Manager Class. Handles creating tracker process and relying positions back
    """

    def __init__(self) -> None:
        self._tracking = mp.Event()
        self._tracking_process = None

        # Share positions between processes
        self._manager = mp.Manager()
        self._positions = self._manager.list()
        self._pos_lock = self._manager.Lock()
        self._velocities = self._manager.list()

    def start_tracking_objects(
        self, device: int, spheros: int, init_positions: list = []
    ):
        """
        Start tracking process
        """
        self._tracking.set()

        self._tracking_process = mp.Process(
            target=SpheroTracker.start_tracking,
            args=(
                device,
                spheros,
                self._tracking,
                self._positions,
                self._pos_lock,
                self._velocities,
                init_positions,
            ),
        )
        self._tracking_process.daemon = True
        self._tracking_process.start()

    def get_positions(self):
        with self._pos_lock:
            try:
                pos = self._positions.pop()
                return pos
            except:
                return None

    def set_velocities(self, velocities):
        with self._pos_lock:
            for i, velocity in enumerate(velocities):
                if i >= len(self._velocities):
                    self._velocities.append(velocity)
                else:
                    self._velocities[i] = velocity

    def cleanup(self):
        self._tracking.clear()
        if self._tracking_process:
            self._tracking_process.join()
        print("Finished tracking process")

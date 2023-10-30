import cv2
from ._video_stream import VideoStream
import multiprocessing as mp
from ._sort import Sort
import numpy as np

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
        spheros: int,
        tracking: mp.Event,
        positions,
        lock,
        init_positions: list = [],
    ):
        self._spheros = spheros
        self._tracking = tracking
        self._positions = positions
        self._lock = lock

        self._sort_tracker = Sort()

        if len(init_positions) > 0:
            pos = np.array(init_positions)

            num_rows = pos.shape[0]
            new_column = np.ones((num_rows, 1))
            pos = np.hstack((pos, new_column))

            self._sort_tracker.update(pos)

        self._stream = VideoStream(0).start()

        # Get scale factors
        print("Tracker: Initalizing frame")
        frame = self._stream.read()
        while frame is None:
            frame = self._stream.read()
            
        self._height, self._width = frame.shape[:2]

        self._scale_factor = 2.0 / min(self._width, self._height)

        print("Tracker: Initalized Frame")

        # It takes some time for the camera to focus, etc



    """
    Private methods
    """

    def _track_objects(self):
        """
        Track objects
        """

        while self._tracking.is_set():
            frame = self._stream.read()
            thresh = self._process_frame(frame)

            dets = self._detect_objects(thresh)

            active_tracks = self._sort_tracker.update(dets)

            pos = self._update_positions(active_tracks)

            sorted_indices = np.argsort(pos[:, 2])

            with self._lock:
                if len(self._positions) >= MAX_LEN:
                    self._positions.pop(0)
                self._positions.append(pos[sorted_indices])

            for track in active_tracks:
                cv2.rectangle(
                    frame,
                    (int(track[0]), int(track[1])),
                    (int(track[2]), int(track[3])),
                    color=(0, 255, 0),
                    thickness=2,
                )

                cv2.putText(
                    frame,
                    "#{}".format(track[4]),
                    (int(track[0]), int(track[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (36, 255, 12),
                    2,
                )


            if (len(active_tracks) < self._spheros):
                print("Not enough spheros detected")
                print([len(active_tracks), len(dets)])
                while True:
                    cv2.imshow("Frame", frame)
                    cv2.imshow("Thresh", thresh)

                    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit the loop
                        break

            # if (len(active_tracks) < 15):
            #     print([len(active_tracks), len(dets)])
            #     while True:
            #         cv2.imshow("Frame", frame)
            #         cv2.imshow("Thresh", thresh)

            #         if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit the loop
            #             break

            cv2.imshow("Frame", frame)
            cv2.imshow("Thresh", thresh)

            # self._writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit the loop
                break

        print("Saving video...")
        cv2.destroyAllWindows()  # Close all OpenCV windows
        self._stream.stop()  # Stop the video stream
        self._out.release()

    def _detect_objects(self, frame):
        """
        Detect Spheros in a frame

        Args:
            frame: The frame to detect Spheros in

        Returns:
            A list of detections
        """
        contours = find_all_contours(frame)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        endIndex = min(self._spheros, len(contours))

        dets = []
        for contour in contours[:endIndex]:
            x, y, w, h = cv2.boundingRect(contour)
            det = np.array([x, y, x + w, y + h, 1.0])
            dets.append(det)

        return np.array(dets)

    def _update_positions(self, tracks):
        """
        Update the positions of the spheros

        Args:
            dets: The detections to update the positions with
        """

        pos = np.empty((len(tracks), 3))

        for i, track in enumerate(tracks):
            center_x, center_y = self._normalize_coordinates(
                (track[0] + track[2]) / 2, (track[1] + track[3]) / 2
            )

            pos[i] = np.array([center_x, center_y, track[4]])

        return pos

    def _normalize_coordinates(self, x, y):
        normalized_x = self._scale_factor * (x - self._width / 2)
        normalized_y = self._scale_factor * (y - self._height / 2)
        return normalized_x, -normalized_y

    """
    Static methods
    """

    @staticmethod
    def start_tracking(
        spheros: int, tracking: mp.Event, positions, lock, init_positions: list = []
    ):
        """
        Start tracking spheros
        """
        print("Starting to track!")
        sphero_tracker = SpheroTracker(
            spheros, tracking, positions, lock, init_positions
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

        processed_frame = cv2.threshold(processed_frame, 40, 255, cv2.THRESH_BINARY)[1]
        processed_frame = cv2.erode(processed_frame, None, iterations=12)
        processed_frame = cv2.dilate(processed_frame, None, iterations=8)
        # processed_frame = cv2.erode(processed_frame, None, iterations=8)
        # processed_frame = cv2.dilate(processed_frame, None, iterations=12)

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

    def start_tracking_objects(self, spheros: int, init_positions: list = []):
        """
        Start tracking process
        """
        self._tracking.set()

        self._tracking_process = mp.Process(
            target=SpheroTracker.start_tracking,
            args=(
                spheros,
                self._tracking,
                self._positions,
                self._pos_lock,
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

    def cleanup(self):
        self._tracking.clear()
        if self._tracking_process:
            self._tracking_process.join()
        print("Finished tracking process")

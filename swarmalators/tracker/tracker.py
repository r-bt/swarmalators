import cv2
from ._video_stream import VideoStream
import multiprocessing as mp
from ._sort import Sort
from ._tracker_filters import ColorFilter
import numpy as np
import atexit

class Tracker:
    """
    Tracks Spheros in separate process

    Attributes:
        num_spheros: The number of Spheros to track
    """

    def __init__(self, num_spheros: int) -> None:
        self._tracking = mp.Event()
        self._tracking_process = None

        # Number of objects to track
        self._trackable_objects = num_spheros

        # Scale values (filled in later)
        self.scale_factor = None
        self.width = None
        self.height = None

    def start_tracking_objects(self, init_positions: list = []):
        """
        Start tracking process
        """
        self._tracking.set()

        self._tracking_process = mp.Process(target=self._track_objects, args=(init_positions,))
        self._tracking_process.daemon = True
        self._tracking_process.start()
    
    def find_single_sphero(self):
        """
        Find the location of a single (lit) sphero

        Assumes:
            Only one Sphero's LED matrix is on

        Returns:
            The bounding box of the sphero
        """
        # Attempt this 10 times
        box = None
        stream = VideoStream(0).start()

        for i in range(0, 75):
            frame = stream.read()

            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed_frame = cv2.GaussianBlur(processed_frame, (21, 21), 0)
            processed_frame = cv2.threshold(processed_frame, 237, 255, cv2.THRESH_BINARY)[1]
            processed_frame = cv2.dilate(processed_frame, None, iterations=12)

            contours = self._find_all_contours(processed_frame)

            if (len(contours) == 0):
                continue

            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            contour = contours[0]

            x, y, w, h = cv2.boundingRect(contour)
            box = [x,y,x+w,y+h]

            # while True:
            #     cv2.rectangle(frame,
            #             (x, y),
            #             (x + w, y + h),
            #             color=(0, 255, 0), thickness=2)
                
            #     cv2.imshow("Frame", frame)
            #     cv2.imshow("Processed", processed_frame)

            #     if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
            #         break

            break
        
        if (box == None):
            # while True:
            #     cv2.imshow("Frame", frame)
            #     cv2.imshow("Processed", processed_frame)

            #     if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
            #         break
            print("Failed to find color")

        return box
    
    def cleanup(self):
        self._tracking.clear()
        if self._tracking_process:
            self._tracking_process.join()
        print("Finished tracking process")

    """
    Private class methods
    """

    def _track_objects(self, init_positions: list = []):
        """
        Track objects
        """
        sort_tracker = Sort()

        if init_positions:
            pos = np.array(init_positions)

            num_rows = pos.shape[0]
            new_column = np.ones((num_rows, 1))
            pos = np.hstack((pos, new_column))
            
            sort_tracker.update(pos)

        stream = VideoStream(0).start()

        # Get scale factors
        frame = stream.read()
        self.height, self.width = frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (self.width, self.height))

        self.scale_factor = 2.0 / min(self.width, self.height)

        self._tracking_process = None

        while self._tracking.is_set():
            
            frame = stream.read()
            thresh = self._process_frame(frame)

            dets = self._detect_objects(thresh)

            active_tracks = sort_tracker.update(dets)

            self._update_positions(active_tracks)

            for track in active_tracks:
                cv2.rectangle(frame,
                    (int(track[0]), int(track[1])),
                    (int(track[2]), int(track[3])),
                    color=(0, 255, 0), thickness=2)
            
                cv2.putText(frame, '#{}'.format(track[4]), (int(track[0]), int(track[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                
            cv2.imshow("Frame", frame)

            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
                break
        
        print("Saving video...")
        cv2.destroyAllWindows()  # Close all OpenCV windows
        stream.stop()  # Stop the video stream
        out.release()
    
    def _detect_objects(self, frame):
        """
        Detect Spheros in a frame

        Args:
            frame: The frame to detect Spheros in

        Returns:
            A list of detections
        """
        contours = self._find_all_contours(frame)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        endIndex = min(self._trackable_objects, len(contours))

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
            center_x, center_y = self._normalize_coordinates((track[0] + track[2]) / 2, (track[1] + track[3]) / 2)

            pos[i] = np.array([center_x, center_y, track[4]])
        
        return pos

    def _normalize_coordinates(self, x, y):
        normalized_x = self.scale_factor * (x - self.width / 2)
        normalized_y = self.scale_factor * (y - self.height / 2)
        return normalized_x, -normalized_y

    """
    Static methods
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

        processed_frame = cv2.threshold(processed_frame, 140, 255, cv2.THRESH_BINARY)[1]
        processed_frame = cv2.erode(processed_frame, None, iterations=8)
        processed_frame = cv2.dilate(processed_frame, None, iterations=12)

        return processed_frame

    

            


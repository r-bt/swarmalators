import cv2
from threading import Thread

class VideoStream:
    """A CV2 VideoStream wrapper for threading.

    Attributes:
        device: The device index of the camera to use.
    """

    def __init__(self, device):
        self._stream = cv2.VideoCapture(device)

        if not self._stream.isOpened():
            self._stream.open(device)

        (self._grabbed, self._frame) = self._stream.read()

        self._stopped = False

    def start(self):
        """Start the thread to read frames from the video stream."""
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        """Keep looping indefinitely until the thread is stopped."""
        while True:
            if self._stopped:
                return

            (self._grabbed, self._frame) = self._stream.read()
    
    def read(self):
        """Return the current frame."""
        return self._frame
    
    def stop(self):
        """Indicate that the thread should be stopped."""
        self._stopped = True
        # Release the stream
        self._stream.release()
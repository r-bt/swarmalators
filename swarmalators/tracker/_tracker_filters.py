import cv2 
import numpy as np
import colorsys

from .util._color import Color

class BaseFilter():
    def get_mask(self, img):
        return img

class ColorFilter(BaseFilter):
    """
    Creates a filter for a specific color

    Attributes:
        rgb: The RGB color to filter for – r,g,b in [0,255]
    """

    max_deg_value = 180.0

    def __init__(self, rgb):
        
        hsv = colorsys.rgb_to_hsv(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)

        lower_hue = ((hsv[0] * self.max_deg_value) - 10) % self.max_deg_value
        upper_hue = ((hsv[0] * self.max_deg_value) + 10) % self.max_deg_value

        self.lower = np.array([lower_hue, 25, 140], np.uint8)
        self.upper = np.array([upper_hue, 255, 255], np.uint8)

    def get_mask(self, img):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if self.lower[0] < self.upper[0]:
            mask = cv2.inRange(hsv_img, self.lower, self.upper)
        else:
            print("handling")
            # We need to handle cases where we want to filter from 170 degrees to 10 degress since inRange 
            # doesn't handle wrapping around the 180 degree mark
            mask1 = cv2.inRange(hsv_img, self.lower, np.array([self.max_deg_value, self.upper[1], self.upper[2]], np.uint8))
            mask2 = cv2.inRange(hsv_img, np.array([0, self.lower[1], self.lower[2]], np.uint8), self.upper)
            mask = cv2.bitwise_or(mask1, mask2)

        return mask

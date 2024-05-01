from typing import Tuple
from multiprocessing import Queue
from scipy import ndimage
import numpy as np
import cv2
from time import sleep
import time
from matplotlib import pyplot as plt
from dataclasses import dataclass

class Observer:
    def __init__(self, dyn_locs: Queue, observations: Queue, team: str):
        """
        :param dyn_locs (Queue): Queue to store the dynamic locations.
        :param observations (Queue): Queue to store the observations.
        :param team (str): Team color.
        """
        self.observations = observations
        self.dyn_locs = dyn_locs

        # Class specific variables
        self.NAN_ = 2.0
        self.mask_offset_ = 106
        self.crop_rect_ = (70, 300, 115, 530)
        self.right_angle_threshold_ = 10
        self.team_ = team
        
        # HSV range for yellow color
        self.lower_yellow_ = np.array([20, 100, 100], dtype=np.uint8)
        self.upper_yellow_ = np.array([30, 255, 255], dtype=np.uint8)
        self.hsv_threshold_ = 0.5

    def run(self) -> bool:
        """
        Run the observer process to detect dynamic locations.

        :return: bool: True if the process ran successfully.
        """
        while True:
            sleep(1)
            if not self.observations.empty():
                self.observations.empty()

                # Clear the dynamic locations queue
                self.dyn_locs.empty()
                while not self.observations.empty():
                    sleep(0.1)
                    observation = self.observations.get()
                    depth = observation.mid_depth
                    rgb = observation.mid_rgb

                    try:
                        _, cropped_rgb = self.preprocess_frames(depth, rgb)
                    except Exception as e:
                        print(f"[WARNING] Dynamic tracking encountered an error: {e} \n Continuing to next frame...")
                        continue
                    
                    masked_image = self.apply_mask(cropped_rgb)

                    edge_map = self.get_edge_map(masked_image)

                    lines = self.get_lines(edge_map)

                    print ("Number of lines: ", len(lines))

                    right_angle_lines = self.filter_right_angle_lines(lines, masked_image)

                    print ("Number of right angle lines: ", len(right_angle_lines))

                    # Draw the detected lines on the original image 
                    # _, axes = plt.subplots(1, 2, figsize=(10, 6))
                    
                    # for line1, line2 in right_angle_lines:
                    #     x1, y1, x2, y2 = line1[0]
                    #     axes[1].plot([x1, x2], [y1, y2], color='blue', linewidth=2)  
                    #     x1, y1, x2, y2 = line2[0]
                    #     axes[1].plot([x1, x2], [y1, y2], color='blue', linewidth=2)  

                    # axes[0].imshow(edge_map)
                    # axes[1].imshow(cropped_rgb)
                    # plt.show()

                    if len(right_angle_lines) > 0:
                        self.dyn_locs.put(right_angle_lines)

                    print ("Dynamic locations: ", self.dyn_locs.get())

    def filter_right_angle_lines(self, lines: np.ndarray, masked_image: np.ndarray) -> np.ndarray:
        """
        Filter line segments to identify pairs that are approximately orthogonal.

        :param lines (np.ndarray): Detected line segments.

        :return: right_angle_lines (np.ndarray): Filtered right angle line segments.
        """
        right_angle_lines = []
        for line1 in lines:
            x1, y1, x2, y2 = line1[0]
            roi1 = masked_image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
            for line2 in lines:
                # Calculate the angle between the two lines
                angle = np.abs(np.arctan2(line2[0][3] - line2[0][1], line2[0][2] - line2[0][0]) -
                               np.arctan2(line1[0][3] - line1[0][1], line1[0][2] - line1[0][0]))
                angle_deg = np.degrees(angle)

                # Check if the angle difference is close to 90 degrees
                if np.abs(angle_deg - 90) < self.right_angle_threshold_:
                    x1, y1, x2, y2 = line2[0]
                    roi2 = masked_image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
                    if (roi1.shape[0] != 0 and roi1.shape[1] != 0) and (roi2.shape[0] != 0 and roi2.shape[1] != 0):
                        # Check if the region of interest is yellow in color
                        if (self.is_yellow_color(roi1) and self.is_yellow_color(roi2)):
                            right_angle_lines.append((line1, line2))

        return np.array(right_angle_lines)
    
    def preprocess_frames(self, depth_frame: np.ndarray, rgb_frame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the frame to remove noise and improve edge detection.

        :param depth_frame (np.ndarray): Depth frame.
        :param rgb_frame (np.ndarray): RGB frame.

        :return: tuple[np.ndarray, np.ndarray]: Preprocessed depth and RGB frames.
        """
        # TODO: (Satrajit) Kind of inconclusive if these filters are necessary
        # depth = cv2.medianBlur(depth, 5)
        # depth = cv2.morphologyEx(depth, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        depth_frame = np.nan_to_num(depth_frame, nan=self.NAN_)
        cropped_depth = depth_frame[self.crop_rect_[0]:self.crop_rect_[1], self.crop_rect_[2]:self.crop_rect_[3]]
        cropped_rgb = rgb_frame[self.crop_rect_[0]:self.crop_rect_[1], self.crop_rect_[2]:self.crop_rect_[3]]
        cropped_rgb = cv2.cvtColor(cropped_rgb, cv2.COLOR_BGR2RGB)
        # cropped_rgb = cv2.bilateralFilter(cropped_rgb, 9, 45, 60)

        return tuple([cropped_depth, cropped_rgb])
    
    
    def get_edge_map(self, rgb_frame: np.ndarray) -> np.ndarray:
        """
        Get the edge map using the Canny edge detector.

        :param rgb_frame (np.ndarray): RGB frame.

        :return: edge_map (np.ndarray): Edge map.
        """
        grayscale_image = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        return cv2.Canny(grayscale_image, threshold1=50, threshold2=350)
    
    
    def get_lines(self, edge_map: np.ndarray) -> np.ndarray:
        """
        Use the Probabilistic Hough Transform to detect line segments in the edge map.

        :param edge_map (np.ndarray): Edge map.

        :return: lines (np.ndarray): Detected line segments.
        """
        return cv2.HoughLinesP(edge_map, rho=1, theta=np.pi/180, threshold=15, minLineLength=20, maxLineGap=5)
    

    def apply_mask(self, rgb_frame: np.ndarray) -> np.ndarray:
        """
        Apply a mask to get grabbable region of interest.
        Masks half of the image based on current team to remove the non-grabbable area with a parameterized width.

        :param rgb_frame (np.ndarray): Preprocessed RGB frame to apply the mask to.

        :return: masked_image (np.ndarray): Image with mask applied.
        """
        mask = np.ones_like(rgb_frame[:, :, 0])  # Initialize mask with ones (white)

        if self.team_ == "blue":
            # Set the left half of the mask to zero (black)
            mask[:, :rgb_frame.shape[1]//2 + self.mask_offset_] = 0
        else:
            # Set the right half of the mask to zero (black)
            mask[:, rgb_frame.shape[1]//2 - self.mask_offset_:] = 0

        # Multiply the image with the mask to make the right half black
        masked_image = rgb_frame.copy()
        masked_image[mask == 0] = 0  

        return masked_image
    
    
    def is_yellow_color(self, roi: np.ndarray) -> bool:
        """
        Check if the region of interest is yellow in color. 

        :param roi (np.ndarray): Region of interest to check for yellow color.

        :return: bool: True if the region of interest is yellow in color. 
        """
        
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        yellow_mask = cv2.inRange(hsv_roi, self.lower_yellow_, self.upper_yellow_)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        total_pixels = roi.shape[0] * roi.shape[1]
        return yellow_pixels / total_pixels > self.hsv_threshold_    
    
from multiprocessing import Queue
from scipy import ndimage
import numpy as np
import cv2
from time import sleep
import time
import matplotlib
from matplotlib import pyplot as plt
from dataclasses import dataclass

@dataclass
class Block:
    loc: np.ndarray
    life: int
    pair: list

class Observer:
    def __init__(self, dyn_locs: Queue, observations: Queue):
        self.observations = observations
        self.dyn_locs = dyn_locs

    def run(self):
        counter = 0
        plt_colors = ["red",
                      "blue",
                      "purple",
                      "cyan",
                      "black",
                      "brown",
                      "pink",
                      "orange",
                      "white",
                      "white",
                      "white"]
        blocks = []
        while True:
            sleep(1)
            if not self.observations.empty():
                self.observations.empty()
            while self.observations.empty():
                sleep(0.1)
            observation = self.observations.get()
            observation_time = time.time()
            depth = observation.mid_depth
            rgb = observation.mid_rgb
            if depth is None:
                print("depth is None")
                continue

            depth = np.nan_to_num(depth, nan=2.0)
            cropped = depth[70:300, 100:535].copy()
            cropped_rgb = rgb[70:300, 100:535].copy()
            threshold = 1.4
            cropped[cropped > threshold] = 2.0
            gradient_x = ndimage.sobel(cropped, axis=0)
            gradient_y = ndimage.sobel(cropped, axis=1)
            edges = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
            edges = edges > 0.1

            lines = cv2.HoughLinesP((edges * 255).astype(np.uint8), 1, np.pi / 180, threshold=15, minLineLength=20,
                                    maxLineGap=5)

            output_image = cropped_rgb.copy()
            output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

            filtered_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if len(filtered_lines) == 0:
                    filtered_lines.append(line)
                else:
                    unique = True
                    for i, filtered_line in enumerate(filtered_lines):
                        x1_, y1_, x2_, y2_ = filtered_line[0]
                        if np.abs(x1 - x1_) < 20 and np.abs(y1 - y1_) < 20 and np.abs(x2 - x2_) < 20 and np.abs(
                                y2 - y2_) < 20:
                            unique = False
                            break
                    if unique:
                        filtered_lines.append(line)

            horizontal_lines = []
            vertical_lines = []
            for line in filtered_lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if length > 70:
                    continue
                angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
                if angle < 25 and angle > -25:
                    mid_y = (y1 + y2) / 2
                    if mid_y > 160:
                        continue
                    horizontal_lines.append(line)
                elif angle < -75 or angle > 75:
                    mid_x = (x1 + x2) / 2
                    if mid_x < 25 or mid_x > 420:
                        continue
                    vertical_lines.append(line)

            horizontal_mids = []
            for line in horizontal_lines:
                x1, y1, x2, y2 = line[0]
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                width = np.abs(x2-x1)
                horizontal_mids.append([mid_x, mid_y, width])

            # find pairs of parallel vertical lines that are a certain distance from each other
            # store as pairs of lines
            vertical_pairs = []
            found = []
            for i, line1 in enumerate(vertical_lines):
                if i in found:
                    continue
                x1, y1, x2, y2 = line1[0]
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                for j, line2 in enumerate(vertical_lines):
                    if i == j or j in found:
                        continue
                    x1_, y1_, x2_, y2_ = line2[0]
                    mid_x_ = (x1_ + x2_) / 2
                    mid_y_ = (y1_ + y2_) / 2
                    diff_x = np.abs(mid_x - mid_x_)
                    diff_y = np.abs(mid_y - mid_y_)
                    if diff_y < 20 and 30 < diff_x < 80:
                        left_x = min(mid_x, mid_x_)
                        right_x = max(mid_x, mid_x_)
                        y = (mid_y + mid_y_) / 2
                        for mids in horizontal_mids:
                            if left_x < mids[0] < right_x and 5 < (y - mids[1]) < 40 and (1.0*mids[2]/diff_x) > 0.5:
                                vertical_pairs.append([line1, line2])
                                found.append(i)
                                found.append(j)
                                break


            plt.figure(figsize=(9, 5))
            plt.imshow(output_image)
            for block in blocks:
                block.life -= 1

            for line in horizontal_lines:
                x1, y1, x2, y2 = line[0]
                plt.plot([x1, x2], [y1, y2], color="green", linewidth=2)

            for pair in vertical_pairs:
                x1_1, y1_1, x2_1, y2_1 = pair[0][0]
                x1_2, y1_2, x2_2, y2_2 = pair[1][0]

                top_1 = min(y1_1, y2_1)
                top_2 = min(y1_2, y2_2)

                mid_y = (top_1 + top_2) / 2
                mid_x = (x1_1 + x2_1 + x1_2 + x2_2) / 4
                coords = np.array([mid_x, mid_y])

                found = False
                for block in blocks:
                    dist = np.linalg.norm(block.loc - coords)
                    if dist < 20:
                        found = True
                        block.loc = np.array([mid_x, mid_y])
                        block.life = 20
                        block.pair = pair
                        break
                if not found:
                    blocks.append(Block(np.array([mid_x, mid_y]), 20, pair))


            to_remove = []
            for i, block in enumerate(blocks):
                if block.life <= 0:
                    to_remove.append(i)
            for i, idx in enumerate(to_remove):
                blocks.pop(idx - i)


            # print("Vertical pairs found:", len(vertical_pairs))
            # for i, pair in enumerate(vertical_pairs):
            #     x1, y1, x2, y2 = pair[0][0]
            #     plt.plot([x1, x2], [y1, y2], color=plt_colors[i], linewidth=2, label=f"Cube {i}")
            #     x1, y1, x2, y2 = pair[1][0]
            #     plt.plot([x1, x2], [y1, y2], color=plt_colors[i], linewidth=2)

            for i, block in enumerate(blocks):
                pair = block.pair
                alpha = block.life / 20.0
                x1, y1, x2, y2 = pair[0][0]
                plt.plot([x1, x2], [y1, y2], color=plt_colors[i], linewidth=2, label=f"Cube {i}", alpha=alpha)
                x1, y1, x2, y2 = pair[1][0]
                plt.plot([x1, x2], [y1, y2], color=plt_colors[i], linewidth=2, alpha=alpha)
                x, y = block.loc
                plt.scatter(x, y, color=plt_colors[i], s=100, alpha=alpha)

            end = time.time()
            timing = end - observation_time
            timing = np.round(timing, 2)
            plt.axis('off')
            plt.legend(loc='lower left')
            plt.tight_layout()
            plt.title(f"Frame {counter} - Time: {timing}s - Pairs: {len(vertical_pairs)}")
            plt.savefig(f"plots/output-{counter}.png")
            counter += 1
            plt.close()

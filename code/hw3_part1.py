import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import copy
from utils import video_to_frames, show_single_graph, show_double_graph, spatial_sample


def section1_a():
    # ---------------------------- section 1.a -------------------------------
    # --------------------------------------------- section 1.1 --------------
    pink_floyd_frames, fps = video_to_frames('../given_data/Time_Pink_Floyd.mp4', 33, 33)
    first_frame = pink_floyd_frames[0]
    cv2.imshow("33rd second", first_frame)

    # --------------------------------------------- section 1.2 --------------
    first_frame_copy = copy.deepcopy(first_frame)
    seven_clocks_line = first_frame[292, :, :]
    first_frame_copy[292, :] = [0, 0, 255, 255]
    line_in_blue_channel = seven_clocks_line[:, 0]
    cv2.imshow("33rd second with red line", first_frame_copy)
    show_single_graph(line_in_blue_channel, 'Blue values of the line',
                      'x location in the frame', 'gray level of blue channel')

    # --------------------------------------------- section 1.3 --------------
    sampled_image = spatial_sample(first_frame, 64)
    cv2.imshow("Sampled_line", sampled_image)

    sampled_image_with_red = spatial_sample(first_frame_copy, 64)
    cv2.imshow("Sampled_line", sampled_image_with_red)

    # --------------------------------------------- section 1.4 --------------
    original_sized_red_line_sampled = cv2.resize(sampled_image_with_red, (first_frame.shape[1], first_frame.shape[0]))
    original_sized_sampled = cv2.resize(sampled_image, (first_frame.shape[1], first_frame.shape[0]))
    cv2.imshow("original_sized_sampled", original_sized_red_line_sampled)

    # --------------------------------------------- section 1.5 --------------
    sampled_seven_clocks_line = original_sized_sampled[292, :, :]
    sampled_line_in_blue_channel = sampled_seven_clocks_line[:, 0]
    show_double_graph(line_in_blue_channel, 'Original_graph', sampled_line_in_blue_channel, 'Sampled_graph',
                      'Sampled blue values of the line', 'x location in the frame', 'gray level of blue channel')


def section1_b():
    # ---------------------------- section 1.b -------------------------------
    # --------------------------------------------- section 1.1 --------------
    pink_floyd_frames_b, fps = video_to_frames('../given_data/Time_Pink_Floyd.mp4', 30, 45)

    # --------------------------------------------- section 1.2 --------------
    height, width, layers = pink_floyd_frames_b[0].shape
    size = (width, height)
    original_vid = cv2.VideoWriter('../my_data/original_video_1_b.mp4v', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    sampled_vid = cv2.VideoWriter('../my_data/video_sample_1_b.mp4v', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(len(pink_floyd_frames_b)):
        frame = cv2.resize(spatial_sample(pink_floyd_frames_b[i], 16), size)
        sampled_vid.write(frame)
        original_vid.write(pink_floyd_frames_b[i])
    sampled_vid.release()
    original_vid.release()


if __name__ == '__main__':
    section1_a()
    section1_b()
    cv2.destroyAllWindows()

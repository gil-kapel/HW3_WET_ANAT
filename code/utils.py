import copy

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob


def video_to_frames(vid_path: str, start_second, end_second):
    cap = cv2.VideoCapture(vid_path)
    frame_set = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_time = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    i = 0  # iterator to iterate one frame each loop
    while cap.isOpened():
        ret, frame = cap.read()
        if start_second * fps > i:  # continue until you get to the required start second
            i += 1
            continue
        if end_second * fps < i:  # stop when you get to the required end second
            if end_second == start_second and end_second > 0:
                frame_set.append(cv2.cvtColor(frame, None))
            break
        if not ret or video_time < end_second:
            print('wrong input')
            return
        frame_set.append(cv2.cvtColor(frame, None))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1
    cap.release()
    cv2.destroyAllWindows()
    return np.array(frame_set), fps


def show_single_graph(line, title: str, x_label: str, y_label: str):
    x = np.linspace(0, len(line), len(line))
    plt.plot(x, line)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def show_double_graph(line1, label1, line2, label2,  title: str, x_label: str, y_label: str):
    x = np.linspace(0, len(line1), len(line1))
    plt.plot(x, line1, label=label1)
    plt.plot(x, line2, label=label2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    plt.show()


def spatial_sample_x(first_frame, delta_x):
    mid_line_idx = int(first_frame.shape[1] / 2)
    range_to_sample = int(mid_line_idx/delta_x)
    # sampled_line = np.array([first_frame[:, n * delta_x + mid_line_idx]
    #                          for n in range(-range_to_sample, range_to_sample)])

    sampled_line = []
    original_frame_with_sample_marks = copy.deepcopy(first_frame)
    sampled_line.append(first_frame[:, mid_line_idx])
    original_frame_with_sample_marks[:, mid_line_idx] = [0, 0, 255, 255]
    for n in range(1, range_to_sample):
        sampled_line.append(first_frame[:, n * delta_x + mid_line_idx])
        sampled_line.append(first_frame[:, -n * delta_x + mid_line_idx])
        original_frame_with_sample_marks[:, n * delta_x + mid_line_idx] = [0, 0, 255, 255]
        original_frame_with_sample_marks[:, -n * delta_x + mid_line_idx] = [0, 0, 255, 255]

    sampled_line = np.array(sampled_line)
    sampled_line = np.swapaxes(sampled_line, 0, 1)
    return sampled_line, original_frame_with_sample_marks


def calc_mse(x, x_hat):
    return ((x_hat - x) ** 2).mean(axis=0).mean(axis=0)

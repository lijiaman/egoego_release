import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import numpy as np
import cv2

def assemble_videos(videos, grid_size, description, out_file_name, text_color = (255, 255, 255)):
    x_grid_num = grid_size[1]
    y_grid_num = grid_size[0]
    y_shape, x_shape, _ = videos[0][0].shape
    canvas = np.zeros((y_shape * y_grid_num, x_shape * x_grid_num, 3)).astype(np.uint8)
    

    out = cv2.VideoWriter(out_file_name, cv2.VideoWriter_fourcc(*'FMP4'), 30, (canvas.shape[1], canvas.shape[0]))
    for i in range(len(videos[0])):
        for x in range(x_grid_num):
            for y in range(y_grid_num):
                curr_image = videos[x * y + x][i]
                curr_discription = description[x * y + x]
                canvas[y_shape * y : y_shape * (y + 1),x_shape * x:x_shape * (x + 1), :] = curr_image
                cv2.putText(canvas, curr_discription , (x_shape * x, y_shape * y + 20), 2, 0.5, text_color)
        out.write(canvas)
    out.release()

def crop_center(img,cropx,cropy):
    y,x, _ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx, :]


def crop_side(img,cropx,cropy):
    y,x, _ = img.shape
    startx = x//8-(cropx//2)
    starty = y//2-(cropy//2) 
    return img[starty:starty+cropy,startx:startx+cropx, :]


def read_video_frames(vid_dir):
    cap = cv2.VideoCapture(vid_dir)
    frames = []
    while(cap.isOpened()):
      # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            frames.append(frame)
            pass
        else: 
            break
    cap.release()
    return frames

def write_individaul_frames(frames, output_dir):
    for i in range(len(frames)):
        cv2.imwrite(os.path.join(output_dir, "frame%06d.jpg"%i), frames[i])

def write_frames_to_video(frames, out_file_name = "output.mp4", frame_rate = 30, add_text = None, text_color = (255, 255, 255)):
    y_shape, x_shape, _ = frames[0].shape
    out = cv2.VideoWriter(out_file_name, cv2.VideoWriter_fourcc(*'FMP4'), frame_rate, (x_shape, y_shape))
    transform_dtype = False
    transform_256 = False

    if frames[0].dtype != np.uint8:
        transform_dtype = True
    if np.max(frames[0]) < 1:
        transform_256 = True

    for i in range(len(frames)):
        curr_frame = frames[i]

        if transform_256:
            curr_frame = curr_frame * 256
        if transform_dtype:
            curr_frame = curr_frame.astype(np.uint8)
        if not add_text is None:
            cv2.putText(curr_frame, add_text , (0,  20), 3, 1, text_color)

        out.write(curr_frame)
    out.release()

def read_img_dir(img_dir):
    images = []
    for img_path in sorted(glob.glob(osp.join(img_dir, "*"))):
        images.append(cv2.imread(img_path))
    return images

def read_img_list(img_list):
    images = []
    for img_path in img_list:
        images.append(cv2.imread(img_path))
    return images

def resize_frames(frames, size_x = 224, size_y = 224):
    new_frames = []
    for i in range(len(frames)):
        curr_frame = frames[i]
        curr_frame = cv2.resize(curr_frame, (size_x, size_y))
        new_frames.append(curr_frame)
    return new_frames
    


import torch
import numpy as np
import cv2
import subprocess
import ffmpeg
import os
from matplotlib import pyplot as plt
import sys

class utils:
    @classmethod
    def display_pics(cls, pics, shape=[2,4], color=None):
        fig = plt.figure(figsize=(12,8))
        for i in range(len(pics)):
            plt.subplot(shape[0],shape[1],i+1)
            if color is not None:
                plt.imshow(pics[i], color)
            else:
                plt.imshow(pics[i])
        plt.show()

    @classmethod
    def draw_square(cls, point, img, bgr=False):
        for i in range(-4,5):
            for j in range(-4,5):
                cur_point_y,cur_point_x = round(point[1])+j,round(point[0])+i
                if cls.in_bounds(img.shape, [cur_point_x,cur_point_y]):
                    if bgr:
                        img[cur_point_y,cur_point_x,:] = [0.,0.,1.]
                    else:
                        img[cur_point_y,cur_point_x] = 1.
        return img

    @classmethod
    def in_bounds(cls, image_shape, point):
        if point[0] >= 0 and point[0] < image_shape[1] and point[1] >= 0 and point[1] < image_shape[0]:
            return True
        return False
    
    @classmethod
    def draw_squares(cls, points, img, bgr=False):
        points = cls.to_numpy(points)
        points = points.reshape(-1,2)
        for point in points:
            img = cls.draw_square(point, img, bgr)
        return img
    
    @classmethod
    def get_images_random(cls, num_images, data_dir, out_dir):
        one_liner = "ffmpeg -i in.mp4 -vf select='between(n\,x\,y)' -vsync 0 image%d.png"
        command_args = one_liner.split(' ')
        trial_ids = os.listdir(data_dir)
        cls.process_dir_list(trial_ids)
        trial_paths = list(map(lambda x: os.path.join(data_dir, x), trial_ids))
        for i in range(len(trial_paths)):
            trial_id = trial_ids[i]
            videos = os.listdir(trial_paths[i])
            cls.process_dir_list(videos)
            video_paths = list(map(lambda x: os.path.join(trial_paths[i], x), videos))
            video_num_frames = []
            for j in range(len(videos)):
                metadata = ffmpeg.probe(video_paths[j])
                for stream in metadata['streams']:
                    if stream['codec_type'] == 'video':
                        video_num_frames.append(stream['nb_frames'])
            min_frames = video_num_frames[0]
            for j in range(1, len(videos)):
                min_frames = min(min_frames,video_num_frames[j])
            frames = np.random.rand(num_images+2)*float(min_frames)
            frames_list = frames.tolist()
            indices = list(map(lambda x: round(x), frames_list))
            for idx in indices[1:-1]:
                cur_command_args = command_args.copy()
                cur_dir = os.path.join(out_dir, trial_id+str(idx))
                if not os.path.exists(cur_dir):
                    os.mkdir(cur_dir)
                for j in range(len(videos)):
                    cur_command_args = command_args.copy()
                    cur_command_args[2] = video_paths[j]
                    out_path = os.path.join(cur_dir, videos[j].split('.')[0] + "-" + cur_command_args[-1])
                    cur_command_args[-1] = out_path
                    if os.path.exists(out_path):
                        os.remove(out_path)
                    cur_command_args[4] = "select='between(n\,"+str(idx)+"\,"+str(idx)+")"
                    subprocess.run(cur_command_args, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
    
    @classmethod
    def get_images_even(cls, num_images, data_dir, out_dir):
        one_liner = "ffmpeg -i in.mp4 -vf select='between(n\,x\,y)' -vsync 0 image%d.png"
        command_args = one_liner.split(' ')
        trial_ids = os.listdir(data_dir)
        cls.process_dir_list(trial_ids)
        trial_paths = list(map(lambda x: os.path.join(data_dir, x), trial_ids))
        for i in range(len(trial_paths)):
            trial_id = trial_ids[i]
            videos = os.listdir(trial_paths[i])
            cls.process_dir_list(videos)
            video_paths = list(map(lambda x: os.path.join(trial_paths[i], x), videos))
            video_num_frames = []
            for j in range(len(videos)):
                metadata = ffmpeg.probe(video_paths[j])
                for stream in metadata['streams']:
                    if stream['codec_type'] == 'video':
                        video_num_frames.append(stream['nb_frames'])
            min_frames = video_num_frames[0]
            for j in range(1, len(videos)):
                min_frames = min(min_frames,video_num_frames[j])
            frames = np.linspace(0.,float(min_frames),num_images+2)
            frames_list = frames.tolist()
            indices = list(map(lambda x: round(x), frames_list))
            for idx in indices[1:-1]:
                cur_command_args = command_args.copy()
                cur_dir = os.path.join(out_dir, trial_id+str(idx))
                if not os.path.exists(cur_dir):
                    os.mkdir(cur_dir)
                for j in range(len(videos)):
                    cur_command_args = command_args.copy()
                    cur_command_args[2] = video_paths[j]
                    out_path = os.path.join(cur_dir, videos[j].split('.')[0] + "-" + cur_command_args[-1])
                    cur_command_args[-1] = out_path
                    if os.path.exists(out_path):
                        os.remove(out_path)
                    cur_command_args[4] = "select='between(n\,"+str(idx)+"\,"+str(idx)+")"
                    subprocess.run(cur_command_args, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)

    @classmethod
    def get_n_images_per_video(cls, num_images, data_dir, out_dir, method="even"):
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if method == "even":
            cls.get_images_even(num_images, data_dir, out_dir)
        elif method == "random":
            cls.get_images_random(num_images, data_dir, out_dir)
        else:
            print("Invalid Method Keyword")
            sys.exit(-1)
        
    @classmethod
    def process_dir_list(cls, input_list):
        input_list.sort()
        if input_list[0] == ".DS_Store":
            del input_list[0]
        if input_list[-1] == ".DS_Store":
            del input_list[-1]

    @classmethod
    def check_shape(cls, tensor, shape):
        tensor_size = tensor.size()
        for i in range(len(shape)):
            assert(tensor_size[i] == shape[i])

    @classmethod
    def check_type(cls, tensor, dtype):
        assert(tensor.dtype == dtype)

    @classmethod
    def check_tensor(cls, tensor, shape=None, dtype=None):
        if shape is not None:
            cls.check_shape(tensor, shape)
        if dtype is not None:
            cls.check_type(tensor, dtype)

    @classmethod
    def get_base_imgs(cls, cam_names, base_img_dir):
        base_images = []
        for cam_name in cam_names:
            base_img_path = base_img_dir+"camera_"+cam_name+"_base_img.png"
            base_img = cv2.imread(base_img_path)
            base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
            base_img = base_img.astype("float32",copy=False)
            base_img /= np.max(base_img)
            base_images.append(base_img)
        return base_images

    @classmethod
    def to_numpy(cls, thing):
        if isinstance(thing, np.ndarray):
            return thing
        elif torch.is_tensor(thing):
            return thing.detach().cpu().numpy()
        elif isinstance(thing, list):
            return np.array(thing)
        else:
            raise TypeError("Please pass a list, tensor, or ndarray.")
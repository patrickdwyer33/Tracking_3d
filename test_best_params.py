from dataset import TrainingData
from dataset import get_rand_pics
from torch.utils.data import DataLoader
import os
import json
import torch
import numpy as np
import cameras
from utils import utils
from copy import deepcopy
import sys
from projector import project

def get_test_points(center_x, center_y):
    world_center = np.array([center_x, center_y, 39.], dtype=np.double) #center of arena in top cams coord system
    points = [world_center]
    for offset in [(3.0,0,0),
                   (-3.0,0,0),
                   (0,3.0,0),
                   (0,-3.0,0),
                   (0,0,-15)]:
        offset = np.asarray(offset)
        point = world_center + offset
        points.append(point)
    points = np.asarray(points)
    points = torch.from_numpy(points).view(-1,3)
    return points


if __name__ == "__main__":

    base_path = os.path.join(os.getcwd(), "camera_parameters.json")

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            sides = arg.split('=')
            if len(sides) != 2:
                print("Arg: " + arg + "needs an equals character")
            keyword = sides[0].strip()
            val = sides[1].strip()
            if keyword == "in_dirname":
                base_path = os.path.join(os.getcwd(), val)
            elif keyword == "in_dir_absolute":
                base_path = val

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    body_parts_path = os.path.join(os.getcwd(), "body_parts.json")
        
    body_parts_info = None
    with open(body_parts_path, 'r', encoding="utf-8") as f:
        body_parts_info = json.load(f)

    body_parts = body_parts_info["names"]
    edges = body_parts_info["edges"]

    num_body_parts = len(body_parts)

    arena_info_path = os.path.join(os.getcwd(), "arena_info.json")

    arena_info = None
    with open(arena_info_path, 'r', encoding="utf-8") as f:
        arena_info = json.load(f)["names"]

    cams = cameras.get_cameras()

    N_cams = len(cams)

    r = arena_info["Max_Radius"]
    arena_height = arena_info["Arena_Height"]
    # points per inch
    ppi = arena_info["Points_Per_Inch"]
    num_x_points = num_y_points = (r * 2 * ppi) + 1
    num_z_points = (arena_height * ppi) + 1
    # following two vals were painstakingly tested/computed but are still noisy
    # but are actually only need for evaluation
    center_xy = arena_info["Arena_Center_xy_From_Top"]
    center_x = center_xy[0]
    center_y = center_xy[1]

    Training_Data = TrainingData(num_body_parts=num_body_parts)
    data_loader = None
    if not torch.cuda.is_available():
        data_loader = DataLoader(
            Training_Data,
            batch_size = 5,
            #num_workers = num_cpu_cores, # This works on google collab, but I couldn't get it to work on my machine
            shuffle = True
        )
    else:
        data_loader = DataLoader(
            Training_Data,
            batch_size = 5,
            #num_workers = num_cpu_cores, # This works on google collab, but I couldn't get it to work on my machine
            shuffle = True,
            pin_memory = True
        )
    
    world_points = get_test_points().view(-1,1,3)
    pics = utils.to_numpy(get_rand_pics(data_loader)[4:8,:,:])
    proj_pics = deepcopy(pics)

    best_params = None
    with open(base_path+"camera_parameters.json") as f:
        data = json.load(f)
        max_params = data['best_params']
    best_params = max_params

    for i in range(len(cams)):
        rvec = torch.zeros(3,dtype=torch.double,device=device)
        tvec = torch.zeros(3,dtype=torch.double,device=device)
        if i != 1:
            k = i
            if i > 0:
                k -= 1
            rvec = best_params[(k*6):(k*6)+3].clone()
            tvec = best_params[(k*6)+3:(k*6)+6].clone()
        cam = cams[i]
        K = cam.K
        D = cam.D
        proj_points = project(world_points, rvec,tvec,K,D,cam.fisheye)
        proj_pics[i] = utils.draw_squares(proj_points, proj_pics[i])
    to_display = np.concatenate((pics, proj_pics), axis=0)
    utils.display_pics(to_display, shape=[2,4], color="gray")



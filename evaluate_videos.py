import os
import sys
from utils import utils
from model import Model
from dataset import TestData
import torch
import cameras
import json
import numpy as np
from projector import project
from projector import get_projectors
from torch.utils.data import DataLoader
from dataset import get_batch
import cv2
from copy import deepcopy

def create_evaluate_data(data_dir, out_dir, num_images):
    utils.get_n_images_per_video(num_images, data_dir, out_dir, method="random")

if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(), "All_Data")
    out_dir = os.path.join(os.getcwd(), "Evaluated_Data")
    model_path = os.path.join(os.getcwd(), "trained_model.pt")
    create_data = True
    num_images = 1
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            sides = arg.split('=')
            if len(sides) != 2:
                print("Arg: " + arg + "needs an equals character")
            keyword = sides[0].strip()
            val = sides[1].strip()
            if keyword == "in_dirname":
                data_dir = os.path.join(os.getcwd(), val)
            elif keyword == "in_dir_absolute":
                data_dir = val
            elif keyword == "num_images":
                num_images = int(val)
            elif keyword == "out_dirname":
                out_dir = os.path.join(os.getcwd(), val)
            elif keyword == "out_dir_absolute":
                out_dir = val
            elif keyword == "model_path":
                out_dir = os.path.join(os.getcwd(), val)
            elif keyword == "model_path_absolute":
                out_dir = val
            elif keyword == "create_data":
                create_data = bool(val)
            else:
                print("Invalid Keyword!: " + keyword)

    if create_data:
        create_evaluate_data(data_dir, out_dir, num_images)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    cams = cameras.get_cameras()

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

    N_cams = len(cams)

    r = arena_info["Max_Radius"]
    arena_height = arena_info["Arena_Height"]
    # points per inch
    ppi = arena_info["Points_Per_Inch"]
    num_x_points = num_y_points = (r * 2 * ppi) + 1
    num_z_points = (arena_height * ppi) + 1
    # following two vals were painstakingly tested/computed but are still noisy
    # but are actually only need for evaluation
    #center_xy = arena_info["Arena_Center_xy_From_Top"]
    #center_x = center_xy[0]
    #center_y = center_xy[1]
    x, y, z = np.linspace(-r,r,num_x_points), np.linspace(-r,r,num_y_points), np.linspace(0,arena_height,num_z_points)
    inner_radius = 3.0
    outer_radius = 6.693
    xxx, yyy, zzz = np.meshgrid(x,y,z)
    all_points = np.rot90(np.vstack(list(map(np.ravel, [xxx,yyy,zzz]))))
    all_points_tensor = torch.tensor(all_points.copy(),dtype=torch.double,device=device)
    all_image_points = []
    with torch.no_grad():
        for i in range(N_cams):
            cam = cams[i]
            image_points = project(all_points_tensor,
                                torch.tensor(cam.rvec,dtype=torch.double,device=device),
                                torch.tensor(cam.tvec,dtype=torch.double,device=device),
                                K_mx=cam.K,
                                    D_vec=cam.D,
                                fisheye=cam.fisheye
                                )
            all_image_points.append(image_points)

    def bilinear_interp(image, image_points):
        # This is just the first formula from wikipedia for bilinear interpolation expressed in py-torch
        image_x = image_points[:, 0:1].squeeze()
        image_y = image_points[:, 1:2].squeeze()
        lower_bound = torch.zeros_like(image_x)
        upper_bound_y = torch.ones_like(image_x)*1023
        upper_bound_x = torch.ones_like(image_x)*1279 
        x1 = torch.floor(image_x)
        y1 = torch.floor(image_y)
        x2 = x1 + 1
        y2 = y1 + 1
        x1 = torch.max(torch.min(x1, upper_bound_x),lower_bound)
        y1 = torch.max(torch.min(y1, upper_bound_y),lower_bound)
        x2 = torch.max(torch.min(x2, upper_bound_x),lower_bound)
        y2 = torch.max(torch.min(y2, upper_bound_y),lower_bound)
        left = torch.cat(((x2-image_x).reshape(-1,1),(image_x-x1).reshape(-1,1)), axis=1).reshape(-1,1,2)
        y1_idx = y1.to(torch.int).detach().cpu().numpy().tolist()
        y2_idx = y2.to(torch.int).detach().cpu().numpy().tolist()
        x1_idx = x1.to(torch.int).detach().cpu().numpy().tolist()
        x2_idx = x2.to(torch.int).detach().cpu().numpy().tolist()
        middle_top_left = image[y1_idx,x1_idx].reshape(-1,1)
        middle_top_right = image[y2_idx,x1_idx].reshape(-1,1)
        middle_top = torch.cat((middle_top_left,middle_top_right),dim=1).reshape(-1,1,2)
        middle_bot_left = image[y1_idx,x2_idx].reshape(-1,1)
        middle_bot_right = image[y2_idx,x2_idx].reshape(-1,1)
        middle_bot = torch.cat((middle_bot_left,middle_bot_right),dim=1).reshape(-1,1,2)
        middle = torch.cat((middle_top,middle_bot),dim=1)
        right = torch.cat(((y2-image_y).reshape(-1,1),(image_y-y1).reshape(-1,1)), axis=1).reshape(-1,2,1)
        middle_right = torch.matmul(middle, right)
        middle_right = middle_right.reshape(-1,2,1)
        vals = torch.matmul(left, middle_right).reshape(-1)
        return vals

    def idx_converter(point, range_begin, range_end, num_points):
        interval_length = (range_end - range_begin) / (num_points - 1)
        return int((point - range_begin) / interval_length)

    x_indices = []
    y_indices = []
    z_indices = []
    for point in all_points:
        x_idx = idx_converter(point[0], x[0], x[-1], num_x_points)
        y_idx = idx_converter(point[1], y[0], y[-1], num_y_points)
        z_idx = idx_converter(point[2], z[0], z[-1], num_z_points)
        x_indices.append(x_idx)
        y_indices.append(y_idx)
        z_indices.append(z_idx)

    def generate_model_input(images, N_cams=N_cams):
        all_vals = []
        for i in range(N_cams):
            image_points = all_image_points[i]
            image = images[i]
            interpolated_vals = bilinear_interp(image,image_points).to(torch.double).to(device)
            all_vals.append(interpolated_vals)
        vals = torch.zeros(N_cams,len(z),len(y),len(x),dtype=torch.double, device=device)
        for i in range(N_cams):
            vals[i,z_indices,y_indices,x_indices] = all_vals[i]
        return vals

    Test_Data = TestData(transform=generate_model_input, num_body_parts=num_body_parts)

    model = Model(len(z), len(y), len(x))

    model.load_state_dict(torch.load(model_path))

    model = model.to(device)
    model = model.to(torch.double)
    model.eval()
    print(model)

    cams = cameras.get_cameras()
    projectors = get_projectors(cams)

    trial_dirs = os.listdir(out_dir)
    utils.process_dir_list(trial_dirs)
    trial_paths = list(map(lambda x: os.path.join(out_dir, x), trial_dirs))

    N_cams = len(cams)
    
    for i in len(trial_paths):
        trial_path = trial_path[i]
        labels_path = os.path.join(trial_path, "labels.csv")
        if os.path.exists(labels_path):
            os.remove(labels_path)
        pic_names = os.listdir(trial_path)
        utils.process_dir_list(pic_names)
        pic_paths = list(map(lambda x: os.path.join(trial_path, x), pic_names))
        pics = list(map(lambda x: cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2GRAY), pic_paths))
        pics_tensor = Test_Data.__getitem__(i)
        points = model(pics_tensor)
        projected_pic_names = list(map(lambda x: "projected_"+x, pic_names))
        projected_pic_paths = list(map(lambda x: os.path.join(trial_path, x), projected_pic_names))
        projected_pics = deepcopy(pics)
        for j in range(N_cams):
            projected_img = utils.draw_squares(points, projected_pics[j])
            cv2.imwrite(projected_pic_paths[j], projected_img)

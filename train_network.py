import cameras
from utils import utils
import os
import torch
import numpy as np
import json
from projector import project
from projector import get_projectors
from dataset import TrainingData
import multiprocessing
from torch.utils.data import DataLoader
from model import Model
from model import loss_fn as model_loss
import gc
import sys

if __name__ == "__main__":
    in_path = None
    out_path = os.path.join(os.getcwd(), "trained_model.pt")
    n_epochs = 1
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            sides = arg.split('=')
            if len(sides) != 2:
                print("Arg: " + arg + "needs an equals character")
            keyword = sides[0].strip()
            val = sides[1].strip()
            if keyword == "out_path":
                out_path = os.path.join(os.getcwd(), val)
            elif keyword == "out_path_absolute":
                out_path = val
            elif keyword == "n_epochs":
                n_epochs = int(val)
            elif keyword == "in_path":
                in_path = os.path.join(os.getcwd(), val)
            elif keyword == "in_path_absolute":
                in_path = val
            else:
                print("Invalid Keyword!: " + keyword)

    num_cpu_cores = multiprocessing.cpu_count()

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
        arena_info = json.load(f)

    N_cams = len(cams)

    r = arena_info["Max_Radius"]
    arena_height = arena_info["Arena_Height"]
    # points per inch
    ppi = arena_info["Points_Per_Inch"]
    num_x_points = num_y_points = (r * 2 * ppi) + 1
    num_z_points = (arena_height * ppi) + 1
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
                                cam.rvec.clone().detach(),
                                cam.tvec.clone().detach(),
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

    Training_Data = TrainingData(transform=generate_model_input, num_body_parts=num_body_parts)
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

    model = Model(len(z), len(y), len(x))

    if in_path is not None:
        model.load_state_dict(torch.load(in_path))

    model = model.to(device)
    model = model.to(torch.double)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(model)

    projectors = get_projectors(cams)

    loss_fn = lambda x: model_loss(x[0], x[1], projectors) # edges=edges) this is where'd you'd add edges if you measure them

    def train_loop(dataloader, model, loss_fn, optimizer, save_iter=True):
        size = len(dataloader.dataset)
        model.train()
        for idx, batch in enumerate(dataloader):
            if idx > 2 and idx < 45: 
                print(idx)
                continue
            batch_in = batch[0]
            target = batch[1]
            batch_in = batch_in.to(device)
            batch_in = batch_in.to(torch.double)
            pred = model(batch_in)
            loss_input = (pred, target)
            loss = loss_fn(loss_input)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(idx)
            if idx % 100 == 0:
                loss_item, current = loss.item(), (idx + 1) * len(batch_in)
                if save_iter: torch.save(model.state_dict(), out_path)
                print(f"loss: {loss_item:>7f}  [{current:>5d}/{size:>5d}]")
                gc.collect()

    n_epochs = 1
    for t in range(n_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(data_loader, model, loss_fn, optimizer)
    print("Done!")

    torch.save(model.state_dict(), out_path)

    print("Model Saved.")

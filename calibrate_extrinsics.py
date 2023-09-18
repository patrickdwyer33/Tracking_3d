import numpy as np
import torch
from matplotlib import pyplot as plt
import os
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Union, Callable, List, Tuple
import json
import cameras
from dataset import TrainingData
import gc
import sys
from utils import utils
from projector import get_projectors
from torchimize.functions import lsq_lma as torchimize_lsq_lma

def create_dirs(image_points, cams):
    dirs = torch.zeros(image_points.shape[0],len(cams),3, dtype=torch.double, device=device)
    image_points = utils.to_numpy(image_points)
    for i in range(len(image_points)):
        trial_points = image_points[i]
        for j in range(len(cams)):
            cam = cams[j]
            cam_image_points = trial_points[j]
            if cam_image_points[0] != -1.:
                K_mx = cam.K
                D_mx = cam.D
                cam_points = cv2.undistortPoints(cam_image_points.reshape(1,2), K_mx, D_mx).reshape(2)
                cam_points = np.concatenate((cam_points,np.array([1.],dtype=np.double)),axis=0)
                norm = np.linalg.norm(cam_points)
                unit_dir = cam_points / norm
                unit_dir = unit_dir.astype(np.double)
                dirs[i,j] = torch.tensor(unit_dir)
    return dirs

def Compute_Residuals(
        params : torch.Tensor,
        image_points : torch.Tensor,
        dirs : torch.Tensor,
        cams : list
    ) -> torch.Tensor :
    """
    INPUT SHAPES:
    params.shape = [18]
    image_points.shape = [N, 4, 3]
    len(cams) = 4
    OUTPUT SHAPE:
    [2*N] (A vector of truth - pred for x and y pixel locations)
    """
    params = params.view(3, 2, 3)
    truth_points = torch.clone(image_points)
    projectors = get_projectors(cams, params=params)
    num_cams = len(cams)

    triangulated_points = []

    for trial_dirs in dirs:
        As = []
        Bs = []
        for i in range(num_cams):
            unit_dir = trial_dirs[i]
            if unit_dir[0] == 0. and unit_dir[1] == 0. and unit_dir[2] == 0.:
                continue
            padded_unit_dir = torch.cat((unit_dir,torch.tensor([0.],dtype=params.dtype,device=params.device)),dim=0).reshape(1,4)
            half_first_mx = torch.cat((padded_unit_dir, padded_unit_dir, padded_unit_dir),dim=0)
            first_mx = torch.cat((half_first_mx, torch.eye(4,dtype=params.dtype,device=params.device)),dim=0)
            assert first_mx.shape[0] == 7
            assert first_mx.shape[1] == 4
            dir_mx = torch.eye(3,dtype=params.dtype,device=params.device)
            for j in range(3):
                dir_mx[j,j] = unit_dir[j].item()
            bottom_zeros = torch.zeros(4,3,dtype=params.dtype,device=params.device)
            top_zeros = torch.zeros(3,4,dtype=params.dtype,device=params.device)
            second_mx_first_half = torch.cat((dir_mx, top_zeros), dim=1)
            bottom_id = torch.eye(4,dtype=params.dtype,device=params.device)
            second_mx_second_half = torch.cat((bottom_zeros, bottom_id), dim=1)
            second_mx = torch.cat((second_mx_first_half, second_mx_second_half), dim=0)
            third_mx = torch.zeros((4,7),dtype=params.dtype,device=params.device)
            for j in range(3):
                third_mx[j,j] = -1.
                third_mx[j,j+3] = 1.
            third_mx[3,6] = 1.
            tmp_mx = torch.matmul(third_mx, second_mx)
            tmp_mx2 = torch.matmul(tmp_mx, first_mx)

            R = torch.eye(3,3,requires_grad=True,dtype=params.dtype,device=params.device)
            tvec = torch.zeros(3, requires_grad=True, dtype=params.dtype, device=params.device)
            if i != 1:
                k = i
                if i > 0:
                    k -= 1
                rvec = params[k:k+1,0:1,:].clone().squeeze()
                tvec = params[k:k+1,1:,:].clone().squeeze()
                r_norm = torch.linalg.vector_norm(rvec)
                rvec_ = torch.clone(rvec)
                rvec_ /= r_norm
                cos_norm = torch.cos(r_norm)
                R = cos_norm * torch.eye(3,dtype=params.dtype,device=device)
                R += (1-cos_norm)*torch.matmul(rvec_.reshape(3,1),rvec_.reshape(1,3))
                R += torch.sin(r_norm) * torch.tensor([
                    [0., -rvec_[2].item(), rvec_[1].item()],
                    [rvec_[2].item(), 0., -rvec_[0].item()],
                    [-rvec_[1].item(), rvec_[0].item(), 0.]
                ],dtype=params.dtype,device=params.device).reshape(3,3)
            half_T = torch.cat((R,tvec.view(3,1)),dim=1)
            T = torch.cat((half_T, torch.tensor([[0.,0.,0.,1.]],dtype=params.dtype,device=params.device)),dim=0)
            final_mx = torch.matmul(tmp_mx2, T)
            B = torch.clone(final_mx[:,3:4].squeeze())
            A = final_mx[:,:3]
            A = torch.cat((A, torch.tensor([[0.],[0.],[0.],[1.]],dtype=params.dtype,device=params.device)), dim=1)
            As.append(A)
            Bs.append(B)
        A = torch.cat(As, dim=0)
        B = torch.cat(Bs, dim=0).reshape(-1, 1)
        solution_object = torch.linalg.lstsq(A, B, driver='gels')
        solution = solution_object[0].squeeze().to(torch.double)
        #print('---------------')
        #if solution[2] < 0:
        #    print_debug(solution_object, A, B, num_valid, trial_Ts, trial_dirs, trial_points)
        #else:
        #    print(solution_object.residuals)
        solution = solution[:3]
        triangulated_points.append(solution)

    triangulated_points = torch.cat(triangulated_points, dim=0).reshape(-1,3)
    projected_points = []
    for i in range(num_cams):
        projector = projectors[i]
        points = projector(triangulated_points)
        points = points.view(-1, 1, 2)
        projected_points.append(points)

    all_proj_points = torch.cat(projected_points, dim=1)
    assert all_proj_points.shape[1] == 4
    assert all_proj_points.shape[2] == 2
    pred = all_proj_points.reshape(-1)
    truth = truth_points.reshape(-1)
    pred = pred[truth != -1.]
    truth = truth[truth != -1.]
    out = truth - pred
    out = out.squeeze()
    return out


if __name__ == "__main__":

    base_path = os.path.join(os.getcwd(), "camera_params.json")

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

    cams = cameras.get_cameras(extrinsics_grad=True)

    body_parts_path = os.path.join(os.getcwd(), "body_parts.json")
            
    body_parts_info = None
    with open(body_parts_path, 'r', encoding="utf-8") as f:
        body_parts_info = json.load(f)

    body_parts = body_parts_info["names"]
    edges = body_parts_info["edges"]

    num_body_parts = len(body_parts)

    def no_images_transform(images_tensor):
        return torch.tensor([0.],dtype=torch.double,device=device)

    Training_Data = TrainingData(transform=no_images_transform, num_body_parts=num_body_parts)
    data_loader = None
    if not torch.cuda.is_available():
        data_loader = DataLoader(
            Training_Data,
            batch_size = 1,
            #num_workers = num_cpu_cores, # This works on google collab, but I couldn't get it to work on my machine
            shuffle = True
        )
    else:
        data_loader = DataLoader(
            Training_Data,
            batch_size = 1,
            #num_workers = num_cpu_cores, # This works on google collab, but I couldn't get it to work on my machine
            shuffle = True,
            pin_memory = True
        )

    all_labels = []
    for idx, batch in enumerate(data_loader):
        labels = batch[1].squeeze()
        gc.collect()
        labels = labels.t()
        labels = labels.reshape(8, 4, 2)
        all_labels.append(labels)
    
    image_points = torch.cat(all_labels, dim=0)
    image_points = image_points.to(torch.double).to(device)

    taus = np.linspace(1e-1, 1e-5, 50)

    image_shape = np.array([1024,1280])

    best_params = []
    data = None
    with open(base_path, 'r') as f:
        data = json.load(f)
        counter = data['counter']
        max_params = data['best_params']

    payload = (torch.tensor(max_params[0],dtype=torch.float,device=device).squeeze(),
                torch.tensor(max_params[1],dtype=torch.float,device=device).squeeze())
    best_params.append(payload)

    N_cams = len(cams)
    Ts = []
    for i in range(N_cams):
        if i == 1: continue
        T = cams[i].T.reshape(1, 4, 4)
        Ts.append(T)
    
    for_param_Ts = torch.cat(Ts, dim=0)

    Rmx1 = for_param_Ts[0,:3,:3]
    Rmx2 = for_param_Ts[1,:3,:3]
    Rmx3 = for_param_Ts[2,:3,:3]
    Rmx1 = utils.to_numpy(torch.clone(Rmx1)).reshape(3,3)
    Rmx2 = utils.to_numpy(torch.clone(Rmx2)).reshape(3,3)
    Rmx3 = utils.to_numpy(torch.clone(Rmx3)).reshape(3,3)
    Rvec1 = torch.tensor(cv2.Rodrigues(Rmx1)[0], dtype=torch.double, device=device).view(1,3)
    Rvec2 = torch.tensor(cv2.Rodrigues(Rmx2)[0], dtype=torch.double, device=device).view(1,3)
    Rvec3 = torch.tensor(cv2.Rodrigues(Rmx3)[0], dtype=torch.double, device=device).view(1,3)
    rvecs = torch.cat((Rvec1,Rvec2,Rvec3), dim=0).view(3,1,3)
    tvecs = torch.cat((for_param_Ts[0,:3,3].view(1,3),for_param_Ts[1,:3,3].view(1,3),for_param_Ts[2,:3,3].view(1,3)), dim=0).view(3,1,3)
    dirs = create_dirs(image_points.clone(), cams)
    
    for k in range(counter, len(taus)):
        tau = taus[k]
        counter += 1
        print(counter)
        for j in range(100):
            rvecs_ = torch.clone(rvecs)
            tvecs_ = torch.clone(tvecs)
            if j != 0:
                rvecs_ = torch.rand(3,1,3, dtype=torch.double, device=device)
                tvecs_ += ((torch.rand(3,1,3, dtype=torch.double, device=device) - 0.5) * 20)
            params = torch.cat((rvecs_,tvecs_),dim=1)
            params = params.view(-1)
            params = params.clone().detach().requires_grad_(True)
            print("inner_idx: " + str(j))
            print("params: ")
            print(params.detach().numpy().tolist())
            args = (image_points, dirs, cams)
            fn_wrapper = lambda x: Compute_Residuals(x, *args)
            extrinsics = torchimize_lsq_lma(params,
                            function=Compute_Residuals,
                            args=args,
                            tau=tau,
                            meth='lev')[-1]
            loss = torch.square(fn_wrapper(extrinsics)).sum()
            best_params.append((loss,extrinsics))
        best_params.sort(key = lambda x: x[0].item())
        with open(base_path, 'w') as f:
            min_stuff = best_params[0]
            new_stuff = [min_stuff[0].detach().cpu().numpy().tolist(),min_stuff[1].detach().cpu().numpy().tolist()]
            data["best_params"] = new_stuff
            data["counter"] = counter
            f.write(json.dumps(data))
            
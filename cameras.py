import os
import numpy as np
import torch
import cv2
import json

class Camera:
    def __init__(self, name, rvec, tvec, K, D, fisheye):
        self.name = name
        self.rvec = rvec
        self.tvec = tvec
        self.K = K
        self.D = D
        rvec_numpy = np.array(rvec,dtype=np.double).flatten()
        self.R = torch.tensor(cv2.Rodrigues(rvec_numpy)[0].reshape(3,3))
        tvec_reshaped = tvec.reshape(3,1)
        top_T = torch.cat((torch.clone(self.R), tvec_reshaped), dim=1)
        bottom_T = torch.tensor([[0.,0.,0.,1.]],dtype=torch.double)
        self.T = torch.cat((top_T, bottom_T), dim=0)
        self.update_cam_pos()
        self.fisheye = fisheye
        
    def set_T(self, new_T):
        self.T = new_T
        self.update_R_mx()
        self.update_R_vec()
        self.update_T_vec()
        self.update_cam_pos()
    
    def update_R_mx(self):
        self.R = self.T[:3,:3]
    
    def update_R_vec(self):
        self.rvec = torch.tensor(cv2.Rodrigues(self.R)[0].flatten(),dtype=torch.double)
        
    def update_T_vec(self):
        self.tvec = self.T[:3,3:4].flatten()
        
    def update_cam_pos(self):
        left = -1 * self.R.T
        self.cam_pos = left @ self.tvec

name1 = "17391290"
name2 = "17391304" # top cam
name3 = "19412282"
name4 = "21340171"
names = [name1,name2,name3,name4]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_cameras(extrinsics_grad = False, names=names, device=device, json_path=os.path.join(os.getcwd(), "camera_params.json")):
    data = None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    cams = []
    for name in names:
        fisheye = True
        if name == "17391304":
            fisheye = False
        cam_info = data["current_params"][name]
        rvec = torch.tensor(cam_info["rvec"], requires_grad=extrinsics_grad, dtype = torch.double, device=device).squeeze()
        tvec = torch.tensor(cam_info["tvec"], requires_grad=extrinsics_grad, dtype = torch.double, device=device).squeeze()
        K_mx = np.array(cam_info["K_mx"], dtype=np.float64).reshape(3,3)
        dvec = np.array(cam_info["dvec"], dtype=np.float64).flatten()
        cam = Camera(name, rvec, tvec, K_mx, dvec, fisheye)
        cams.append(cam)
    return cams


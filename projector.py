from utils import utils
import torch
import numpy as np
import cv2
from copy import deepcopy

class ProjectFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, R_vec, T_vec, K_mx, D_vec, fisheye, epsilon):
        out_obj = None
        input_clone = torch.clone(input)
        input_numpy = utils.to_numpy(input_clone)
        num_dims = input_numpy.ndim

        if num_dims == 2:
            input_numpy = input_numpy.reshape(-1,1,3)
        assert input_numpy.ndim == 3

        R_vec_numpy = utils.to_numpy(R_vec)
        T_vec_numpy = utils.to_numpy(T_vec)
        # K_mx should be numpy
        # D_vec should be numpy
        # fisheye should be bool
        # epsilon should be float

        def project_helper(fisheye, cv_inputs):
            out_obj = None
            if fisheye:
                out_obj = cv2.fisheye.projectPoints(*cv_inputs)
            else:
                out_obj = cv2.projectPoints(*cv_inputs)
            return out_obj

        cv_inputs = [input_numpy, R_vec_numpy, T_vec_numpy, K_mx, D_vec]
        out_obj = project_helper(fisheye, cv_inputs)

        out = out_obj[0].reshape(-1, 2)
        proj_jacobian = out_obj[1][:,:6]

        proj_jacobian_tensor = torch.tensor(proj_jacobian, dtype=input.dtype, device=input.device)

        grad_input_list = []

        def estimate_grad(dim, epsilon, fisheye, cv_inputs):
            cv_inputs_minus = deepcopy(cv_inputs)
            cv_inputs_plus = deepcopy(cv_inputs)
            cv_inputs_minus[0][:,:,dim] -= epsilon
            cv_inputs_plus[0][:,:,dim] += epsilon
            out_minus = torch.tensor(project_helper(fisheye, cv_inputs_minus)[0], dtype=input.dtype, device=input.device)
            out_plus = torch.tensor(project_helper(fisheye, cv_inputs_plus)[0], dtype=input.dtype, device=input.device)
            dim_grad = out_plus - out_minus
            dim_grad /= (2*epsilon)
            dim_grad = dim_grad.reshape(-1, 2, 1)
            return dim_grad

        for i in range(3):
            dim_grad = estimate_grad(i, epsilon, fisheye, cv_inputs)
            grad_input_list.append(dim_grad)

        input_jacobian = torch.cat(grad_input_list, dim=2)
        assert input_jacobian.shape[1] == 2
        assert input_jacobian.shape[2] == 3
        assert input_jacobian.dim() == 3

        ctx.save_for_backward(input_jacobian, proj_jacobian_tensor)
        out = torch.tensor(out, requires_grad=True, dtype=input.dtype, device=input.device).reshape(-1,2).squeeze()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_jacobian, proj_jacobian_tensor = ctx.saved_tensors
        grad_input = grad_R_vec = grad_T_vec = grad_K_mx = grad_D_vec = grad_fisheye = grad_epsilon = None
        if ctx.needs_input_grad[0]:
            grad_output_shaped = torch.clone(grad_output).view(-1, 1, 2)
            grad_input = grad_output_shaped.matmul(input_jacobian).squeeze()
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            grad_output_shaped = grad_output_shaped.view(-1, 1)
            grad_params = proj_jacobian_tensor.t().mm(grad_output_shaped).squeeze()
        if ctx.needs_input_grad[1]:
            grad_R_vec = grad_params[:3].squeeze()
        if ctx.needs_input_grad[2]:
            grad_T_vec = grad_params[3:].squeeze()
        return grad_input, grad_R_vec, grad_T_vec, grad_K_mx, grad_D_vec, grad_fisheye, grad_epsilon

def project(input, R_vec=torch.torch.randn(3,dtype=torch.double), T_vec=torch.randn(3,dtype=torch.double), K_mx=np.random.rand(3,3), D_vec=np.zeros(4,dtype=np.double), fisheye=False, epsilon=1e-6):
    return ProjectFunction.apply(input, R_vec, T_vec, K_mx, D_vec, fisheye, epsilon)

def test_projector():
    from torch.autograd import gradcheck
    input = (torch.randn(10,3,dtype=torch.double,requires_grad=True))
    test = gradcheck(project, input, eps=1e-6, atol=1e-4)
    assert(test)
    return test

def get_projectors(cams, params=None):
    projectors = []
    N_cams = len(cams)
    for i in range(N_cams):
        cam = cams[i]
        projector = lambda x: project(x, R_vec=cam.rvec, T_vec=cam.tvec, K_mx=cam.K, D_vec=cam.D, fisheye=cam.fisheye)
        if params is not None:
            for i in range(N_cams):
                cam = cams[i]
                k = i
                if i > 0:
                    k -= 1
                rvec = torch.zeros(3, requires_grad=True, dtype=params.dtype, device=params.device)
                tvec = torch.zeros(3, requires_grad=True, dtype=params.dtype, device=params.device)
                if i != 1:
                    rvec = params[k:k+1,0:1,:].clone().squeeze()
                    tvec = params[k:k+1,1:,:].clone().squeeze()
                fisheye = cam.fisheye
                epsilon = 1e-6
                projector = lambda x: project(x, R_vec=rvec, T_vec=tvec, K_mx=cam.K, D_vec=cam.D, fisheye=cam.fisheye)
        projectors.append(projector)
    return projectors

if __name__ == "__main__":
    print(test_projector())
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def loss_fn(pred, target, projectors, edges=None):
    loss = 0.
    N_cams = len(projectors)
    if edges is not None:
        for edge_info in edges:
            weight = edge_info[3]
            length = edge_info[2]
            dist = (weight * torch.abs(length - torch.square(pred[:,edge_info[0]:edge_info[0]+1,:].squeeze() - pred[:,edge_info[1]:edge_info[1]+1,:].squeeze()))).sum()
            loss += dist
    for i in range(N_cams):
        pred = pred.reshape(-1, 3)
        projector = projectors[i]
        projected = projector(pred).view(-1)
        cur_target = target[:,i*2:(i*2)+2,:].reshape(-1)
        visible_projected = projected[cur_target != -1.]
        visible_cur_target = cur_target[cur_target != -1.]
        dist = torch.square(visible_projected - visible_cur_target).sum()
        loss += dist
    return loss
        

class Model(nn.Module):
    def __init__(self, 
                 input_depth, 
                 input_height, 
                 input_width, 
                 padding=1, 
                 stride=1, 
                 poolstride1=3, 
                 poolstride2=2, 
                 dilation=1, 
                 kernel_size1=4, 
                 kernel_size2=2, 
                 kernel_pool1=3, 
                 kernel_pool2=2):
        super().__init__()
        convs = []
        final_depth = input_depth
        final_height = input_height
        final_width = input_width
        for i in range(4,8):
            conv1 = nn.Conv3d(i,i,kernel_size1,stride=stride,padding=padding)
            final_depth = math.floor(((final_depth + (2*padding) - (dilation * (kernel_size1 - 1)) - 1)/stride)+1)
            final_height = math.floor(((final_height + (2*padding) - (dilation * (kernel_size1 - 1)) - 1)/stride)+1)
            final_width = math.floor(((final_width + (2*padding) - (dilation * (kernel_size1 - 1)) - 1)/stride)+1)
            pool1 = nn.MaxPool3d(kernel_pool1,stride=poolstride1,padding=padding)
            final_depth = math.floor(((final_depth + (2*padding) - (dilation * (kernel_pool1 - 1)) - 1)/poolstride1)+1)
            final_height = math.floor(((final_height + (2*padding) - (dilation * (kernel_pool1 - 1)) - 1)/poolstride1)+1)
            final_width = math.floor(((final_width + (2*padding) - (dilation * (kernel_pool1 - 1)) - 1)/poolstride1)+1)
            conv2 = nn.Conv3d(i,i+1,kernel_size2,stride=stride,padding=padding)
            final_depth = math.floor(((final_depth + (2*padding) - (dilation * (kernel_size2 - 1)) - 1)/stride)+1)
            final_height = math.floor(((final_height + (2*padding) - (dilation * (kernel_size2 - 1)) - 1)/stride)+1)
            final_width = math.floor(((final_width + (2*padding) - (dilation * (kernel_size2 - 1)) - 1)/stride)+1)
            pool2 = nn.MaxPool3d(kernel_pool2,stride=poolstride2,padding=padding)
            final_depth = math.floor(((final_depth + (2*padding) - (dilation * (kernel_pool2 - 1)) - 1)/poolstride2)+1)
            final_height = math.floor(((final_height + (2*padding) - (dilation * (kernel_pool2 - 1)) - 1)/poolstride2)+1)
            final_width = math.floor(((final_width + (2*padding) - (dilation * (kernel_pool2 - 1)) - 1)/poolstride2)+1)
            convs.append(conv1)
            convs.append(pool1)
            convs.append(conv2)
            convs.append(pool2)
        self.conv_net = nn.Sequential(*convs)
        after_conv_num = final_depth*final_height*final_width*8
        self.after_conv_num = after_conv_num
        self.l1 = nn.Linear(after_conv_num, after_conv_num //2)
        self.l2 = nn.Linear(after_conv_num // 2, 8 * 3)
    
    def forward(self, x):
        conv_out = self.conv_net(x).view(-1, self.after_conv_num)
        lin_out = self.l2(F.relu(self.l1(conv_out)))
        out = lin_out.view(-1,8,3)
        return out
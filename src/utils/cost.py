import torch
import torchvision 
from .toy_utils import middle_rgb

def strong_cost(x,y):
    
    """
    x - torch.Size([B,1,H,H])
    y - torch.size([B,1,H,H])
    returns - torch.Size([B,1])
    """
    return torch.norm(x.flatten(start_dim=1,end_dim=-1) - y.flatten(start_dim=1,end_dim=-1),
                      dim=1, keepdim=True).pow(2).mul(0.5)



def cost_image_color_latent(colors_x, y):
    
    """
    colors_x - torch.Size([B,3]) 
    y - torch.Size([B,3,H,W])  - normalize from 0 to 1 
    returns - torch.Size([B,1])
    """
    colors_y = middle_rgb(y) # [B,3]
    
    return torch.norm(colors_x - colors_y, dim=1, p=2,
                      keepdim=True).pow(2).mul(0.5)



def cost_image_shape_latent(x,y):
    
    """
    x - torch.Size([B,1,H,W]) - normalize from 0 to 1
    y - torch.Size([B,3,H,W]) - normalize from 0 to 1
    returns - torch.Size([B,1])
    """
    trnsfm = torchvision.transforms.Grayscale()
    gray_data = trnsfm(y) # [B,1,H,W]
    cost = torch.norm(x.flatten(start_dim=1,end_dim=-1)
                    - gray_data.flatten(start_dim=1,end_dim=-1),p=2,dim=1,keepdim=True).pow(2).mul(0.5)
    
    return cost
 
import torch
import numpy
import matplotlib
from typing import Dict, Any
import os
import pickle
import numpy as np
import torch.nn as nn
import wandb
import gc
from PIL import Image
from typing import Dict, Any
import numpy as np
import random
import torch.distributions as TD
import matplotlib.pyplot as plt


def seed_everything(
    seed: int,
    *,
    avoid_benchmark_noise: bool = False,
    only_deterministic_algorithms: bool = False
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not avoid_benchmark_noise
    torch.use_deterministic_algorithms(only_deterministic_algorithms, warn_only=True)


def sample_gauss(mu, cov, n):
    """
    mu - torch.Size([2])
    cov - torch.Size([2,2])
    n - int (amount of samples)
    """
    dist =  TD.MultivariateNormal(mu, cov)
    return dist.sample(torch.Size([n]))


def plot_initial_data(mus,covs,n):
    """
    mus - list of torch.Size([2])
    covs - list of torch.Size([2,2])
    n - int (amount of samples)
    """
    
    for idx,mu,cov in zip(range(len(mus)), mus,covs):
        d = sample_gauss(mu, cov, n)
        plt.scatter(d[:,0],d[:,1],edgecolor='black',label=f'distribution {idx+1}')
        plt.grid()
        plt.legend()


def fig2img ( fig ):
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )


def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf


#-------- initalization for nets -----#
def normalize_out_to_0_1(x, config):
    #assert torch.min(x) < -0.5
    return torch.clip(0.5*(x+1),0,1) if config.DATASET != 'mnist' else torch.clip(x,-1,1)
 
def init_weights(m):
    """Initialization  for MLP"""
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
def weights_init_D(m):
    """Initialization  for ResNet_D"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
#--------------------------------------#        
def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()    
    
def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)
    

def clean_resources(*tsrs):
    del tsrs
    gc.collect()
    torch.cuda.empty_cache()
    

class Config():

    @staticmethod
    def fromdict(config_dict):
        config = Config()
        for name, val in config_dict.items():
            setattr(config, name, val)
        return config
    
    @staticmethod
    def load(path):
        os.makedirs(os.path.join(*("#" + path).split('/')[:-1])[1:], exist_ok=True)
        with open(path, 'rb') as handle:
            config_dict = pickle.load(handle)
        return Config.fromdict(config_dict)

    def store(self, path):
        os.makedirs(os.path.join(*("#" + path).split('/')[:-1])[1:], exist_ok=True)
        with open(path, 'wb') as handle:
            pickle.dump(self.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def set_attributes(
            self, 
            attributes_dict: Dict[str, Any], 
            require_presence : bool = True,
            keys_upper: bool = True
        ) -> int:
        _n_set = 0
        for attr, val in attributes_dict.items():
            if keys_upper:
                attr = attr.upper()
            set_this_attribute = True
            if require_presence:
                if not attr in self.__dict__.keys():
                    set_this_attribute = False
            if set_this_attribute:
                if isinstance(val, list):
                    val = tuple(val)
                setattr(self, attr, val)
                _n_set += 1
        return _n_set


def get_random_colored_images(images, displace, seed = 0x000000 ):
    np.random.seed(seed)
    
    images = 0.5*(images + 1)
    size = images.shape[0]
    colored_images = []
    hues = 360*np.ones(size)-60
    
    for V, H in zip(images, hues - displace):
        V_min = 0
        
        a = (V - V_min)*(H%60)/60
        V_inc = a
        V_dec = V - a
        
        colored_image = torch.zeros((3, V.shape[1], V.shape[2]))
        H_i = round(H/60) % 6
        
        if H_i == 0:
            colored_image[0] = V
            colored_image[1] = V_inc
            colored_image[2] = V_min
        elif H_i == 1:
            colored_image[0] = V_dec
            colored_image[1] = V
            colored_image[2] = V_min
        elif H_i == 2:
            colored_image[0] = V_min
            colored_image[1] = V
            colored_image[2] = V_inc
        elif H_i == 3:
            colored_image[0] = V_min
            colored_image[1] = V_dec
            colored_image[2] = V
        elif H_i == 4:
            colored_image[0] = V_inc
            colored_image[1] = V_min
            colored_image[2] = V
        elif H_i == 5:
            colored_image[0] = V
            colored_image[1] = V_min
            colored_image[2] = V_dec
        
        colored_images.append(colored_image)
        
    colored_images = torch.stack(colored_images, dim = 0)
  
    
    
    return colored_images


def middle_rgb(x, threshold=0.8):
    
    """
    x - torch.Size([B,3,32,32]) in diapasone [0,1] normalized from Style-GAN and requires_grad
    returns - torch.Size([B,3])
    """
    
    with torch.no_grad():
        # ==== creating_mask ==== 
        x_hsv = matplotlib.colors.rgb_to_hsv(x.permute(0,2,3,1).detach().cpu().numpy()) #[B,32,32,3]
        brightness = torch.from_numpy(x_hsv[:,:,:,-1]) #[B,32,32]
        brightness = brightness.flatten(start_dim=1) #[B, 1024]

        # find indexes with 
        mask = brightness > threshold * brightness.max() # [B,1024]
        #==== creating_mask ====
    
    rgb_image = x.flatten(start_dim=2) # [B,3,1024] : requires_grad
    
    # TODO: rewrite in torch form
    output = torch.zeros(rgb_image.shape[0],3).to(x.device)
    for idx,rgb in enumerate(rgb_image): 
        m = mask[idx]
        output[idx][0] = rgb[0][m].mean()
        output[idx][1] = rgb[1][m].mean() 
        output[idx][2] = rgb[2][m].mean() 
    
    return output


def get_opt_sched(model, lr, total_steps):
    opt = torch.optim.Adam(model.parameters(), lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        lr,
        total_steps=total_steps,
    )
    
    return opt, sched

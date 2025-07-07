import torch
import torchvision.transforms as transforms
import os
import json
import warnings
import numpy as np
import torch
from torch.optim import Optimizer
import torch.nn.functional as F
from torchvision.utils import save_image
import math
from src.data.dataset import build_boundary_distribution

def select_phi(name):
    if name == 'linear':
        def phi(x):
            return x
            
    elif name == 'kl':
        def phi(x):
            return torch.exp(x) - 1
    
    elif name == 'chi':
        def phi(x):
            y = F.relu(x+2)-2
            return 0.25 * y**2 + y
        
    elif name == 'softplus':
        def phi(x):
            return 2*F.softplus(x) - 2*F.softplus(0*x)
    else:
        raise NotImplementedError
    
    return phi
    

class Editor(object):
    def __init__(self, net, image_size, is_cars=False):
        self.img_trans = transforms.Compose([transforms.Resize((256, 256)),])
        self.net = net
        self.generator = net.decoder.cuda()
        self.is_cars = is_cars  # Since the cars StyleGAN output is 384x512, there is a need to crop the 512x512 output.
    
    def run_on_batch(self, inputs):
        images, latents = self.net(inputs, randomize_noise=False, return_latents=True)
        if self.is_cars:
            images = images[:, :, 32:224, :]
        return images, latents

    def _latents_to_image(self, latents):
        # with torch.no_grad():
        images, _ = self.generator([latents], randomize_noise=False, input_is_latent=True)
        if self.is_cars:
                images = images[:, :, 64:448, :]  # 512x512 -> 384x512
            
        return self.img_trans(images)


class EvalAdapted:
    def __init__(self, args, editor):
        # get default configs
        args.train = False
        self.problem_name = args.problem_name
        self.batch_size = args.batch_size
        self.device = args.device
        self.sample_path = f'train_logs/{args.problem_name}/{args.exp}/generated_samples'
        os.makedirs(self.sample_path, exist_ok=True)
        self.editor = editor
        # source/target name
        if args.problem_name.find('_to_') != -1:
            self.source_data_name, self.target_data_name = args.problem_name.split('_to_')
        else:
            self.source_data_name = 'gaussian'
            self.target_data_name = args.problem_name

        # get test sampler
        self.source_sampler, self.target_sampler = build_boundary_distribution(args)

    def generate(self, netE, t, fwd_or_bwd):
        if 'f' == fwd_or_bwd:
            data = self.source_sampler.sample().to(self.device)
        else:
            data = self.target_sampler.sample().to(self.device)
        t = t.to(self.device)
        data2, latent = self.editor.run_on_batch(data)
        diff_latent = netE(t, latent)
        # latent = latent + diff_latent
        latent = self.editor.generator.get_latent(diff_latent)
        generated_data = self.editor._latents_to_image(latent)
        generated_data = (0.5*(generated_data+1)).detach().cpu()
        data = (0.5*(data+1)).detach().cpu()
        
        if 'f' == fwd_or_bwd:
            return (data, data2), generated_data
        else:
            return generated_data, (data, data2)
    

class EvalAdapted2:
    def __init__(self, args, editor):
        # get default configs
        args.train = False
        self.problem_name = args.problem_name
        self.batch_size = args.batch_size
        self.device = args.device
        self.sample_path = f'train_logs/{args.problem_name}/{args.exp}/generated_samples'
        os.makedirs(self.sample_path, exist_ok=True)
        self.editor = editor
        # source/target name
        if args.problem_name.find('_to_') != -1:
            self.source_data_name, self.target_data_name = args.problem_name.split('_to_')
        else:
            self.source_data_name = 'gaussian'
            self.target_data_name = args.problem_name

        # get test sampler
        self.source_sampler, self.target_sampler = build_boundary_distribution(args)

    def generate(self, netE, t, fwd_or_bwd):
        if 'f' == fwd_or_bwd:
            data = self.source_sampler.sample().to(self.device)
        else:
            data = self.target_sampler.sample().to(self.device) 
        t = t.to(self.device)
        data2, latent = self.editor.run_on_batch(data)
        diff_latent = netE(t, latent)
        if 'f' == fwd_or_bwd:
            latent = latent + t[:,None,None] * diff_latent
        else:
            latent = latent + (1 - t[:,None,None]) * diff_latent
        # latent = self.editor.generator.get_latent(diff_latent)
        generated_data = self.editor._latents_to_image(latent)
        generated_data = (0.5*(generated_data+1)).detach().cpu()
        data = (0.5*(data+1)).detach().cpu()
        
        if 'f' == fwd_or_bwd:
            return (data, data2), generated_data
        else:
            return generated_data, (data, data2)


# ------------------------
# EMA
# ------------------------
class EMA(Optimizer):
    def __init__(self, opt, ema_decay):
        '''
        EMA Codes adapted from https://github.com/NVlabs/LSGM/blob/main/util/ema.py
        '''
        self.ema_decay = ema_decay
        self.apply_ema = self.ema_decay > 0.
        self.optimizer = opt
        self.state = opt.state
        self.param_groups = opt.param_groups

    def step(self, *args, **kwargs):
        retval = self.optimizer.step(*args, **kwargs)

        # stop here if we are not applying EMA
        if not self.apply_ema:
            return retval

        ema, params = {}, {}
        for group in self.optimizer.param_groups:
            for i, p in enumerate(group['params']):
                if not p.requires_grad:
                    continue
                state = self.optimizer.state[p]

                # State initialization
                if 'ema' not in state:
                    state['ema'] = p.data.clone()

                if p.shape not in params:
                    params[p.shape] = {'idx': 0, 'data': []}
                    ema[p.shape] = []

                params[p.shape]['data'].append(p.data)
                ema[p.shape].append(state['ema'])

            for i in params:
                params[i]['data'] = torch.stack(params[i]['data'], dim=0)
                ema[i] = torch.stack(ema[i], dim=0)
                ema[i].mul_(self.ema_decay).add_(params[i]['data'], alpha=1. - self.ema_decay)

            for p in group['params']:
                if not p.requires_grad:
                    continue
                idx = params[p.shape]['idx']
                self.optimizer.state[p]['ema'] = ema[p.shape][idx, :]
                params[p.shape]['idx'] += 1

        return retval

    def load_state_dict(self, state_dict):
        super(EMA, self).load_state_dict(state_dict)
        # load_state_dict loads the data to self.state and self.param_groups. We need to pass this data to
        # the underlying optimizer too.
        self.optimizer.state = self.state
        self.optimizer.param_groups = self.param_groups

    def swap_parameters_with_ema(self, store_params_in_ema):
        """ This function swaps parameters with their ema values. It records original parameters in the ema
        parameters, if store_params_in_ema is true."""

        # stop here if we are not applying EMA
        if not self.apply_ema:
            warnings.warn('swap_parameters_with_ema was called when there is no EMA weights.')
            return

        for group in self.optimizer.param_groups:
            for i, p in enumerate(group['params']):
                if not p.requires_grad:
                    continue
                ema = self.optimizer.state[p]['ema']
                if store_params_in_ema:
                    tmp = p.data.detach()
                    p.data = ema.detach()
                    self.optimizer.state[p]['ema'] = tmp
                else:
                    p.data = ema.detach()


class Logger:
    def __init__(self, args, evaltool):
        self.exp_path = f'./train_logs/{args.problem_name}/{args.exp}'
        os.makedirs(self.exp_path, exist_ok=True)
        jsonstr = json.dumps(args.__dict__, indent=4)
        with open(os.path.join(self.exp_path, 'config.json'), 'w') as f:
            f.write(jsonstr)
        with open(os.path.join(self.exp_path, 'log.txt'), 'w') as f:
            f.write("Start Training")
            f.write('\n')

        # self.use_ema = args.use_ema
        self.use_ema = args.use_ema
        self.print_every = args.print_every
        self.save_image_every = args.save_image_every
        self.save_ckpt_every = args.save_ckpt_every
        self.iter = 0
        self.evaltool = evaltool

    def step(self):
        self.iter += 1
        
    def __call__(self, text):
        if (self.iter + 1) % self.print_every == 0:
            with open(os.path.join(self.exp_path, 'log.txt'), 'a') as f:
                f.write(text)
                f.write('\n')

    def save_image(self, info, t):
        if (self.iter+1) % self.save_image_every == 0:
            (real_source, restored_source), generated_target = self.evaltool.generate(info['netE1'], t, 'f')
            generated_source, (real_target, restored_source) = self.evaltool.generate(info['netE2'], t, 'b')
            # save_image(real_source, os.path.join(self.exp_path, f'iter_{self.iter}_real_source.png'))
            # save_image(generated_target, os.path.join(self.exp_path, f'iter_{self.iter}_generated_target.png'))
            # save_image(generated_source, os.path.join(self.exp_path, f'iter_{self.iter}_generated_source.png'))
            # save_image(real_target, os.path.join(self.exp_path, f'iter_{self.iter}_real_target.png'))
            save_image(real_source, os.path.join(self.exp_path, f'real_source.png'))
            save_image(generated_target, os.path.join(self.exp_path, f'generated_target.png'))
            save_image(generated_source, os.path.join(self.exp_path, f'generated_source.png'))
            save_image(real_target, os.path.join(self.exp_path, f'real_target.png'))

    def swap_net(self, info):
        if self.use_ema:
            for key, item in info.items():
                if 'optimizerE' in key:
                    item.swap_parameters_with_ema(store_params_in_ema=True)

    def save_ckpt(self, info, swap=True):
        if (self.iter+1) % self.save_ckpt_every == 0 or self.iter==0:
            if swap:
                self.swap_net(info)
            for key, item in info.items():
                if 'net' in key:
                    torch.save(item.state_dict(), os.path.join(self.exp_path, f'{key}_{self.iter}.pth'))
            if swap:
                self.swap_net(info)


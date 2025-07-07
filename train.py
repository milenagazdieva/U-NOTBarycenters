import sys
sys.path.append('../')
import argparse
import torch
import os
import json
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision
from datetime import datetime
from src.model.ffhq_models.model import Discriminator2, Generator
from src.utils.utils import *
from src.data.dataset import build_boundary_distribution
from argparse import Namespace
from src.model.ffhq_models.psp import pSp

def train(args):
        # save configurations
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    batch_size = args.batch_size

    device = 'cuda:0'
    args.device = device
    
    # Get Data
    args.train = True
    source_dataset, target_dataset = build_boundary_distribution(args)

    model_path = "pretrained_models/e4e_ffhq_encode.pt"
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    # print(opts.start_from_latent_avg)
    net = pSp(opts)
    net.cuda()
    editor = Editor(net, args.image_size)

    # Get Networks/Optimizer
    netD = Discriminator2()
    netG1 = Generator()
    netG2 = Generator()
    
    netG1 = netG1.to(device)
    optimizerG1 = optim.Adam(netG1.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    # if args.use_ema:
    #     optimizerG1 = EMA(optimizerG1, ema_decay=args.ema_decay)
    if args.lr_scheduler:
        schedulerG1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG1, args.num_iterations, eta_min=args.eta_min)
    # netG1 = nn.DataParallel(netG1)

    netG2 = netG2.to(device)
    optimizerG2 = optim.Adam(netG2.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    # if args.use_ema:
    #     optimizerG2 = EMA(optimizerG2, ema_decay=args.ema_decay)
    if args.lr_scheduler:
        schedulerG2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG2, args.num_iterations, eta_min=args.eta_min)
    # netG2 = nn.DataParallel(netG2)

    netD = netD.to(device)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))    
    if args.lr_scheduler:
        schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.num_iterations, eta_min=args.eta_min)
    # netD = nn.DataParallel(netD)

    phi1, phi2 = select_phi(args.phi1), select_phi(args.phi2)

    evaltool = EvalAdapted2(args, editor)
    logger = Logger(args, evaltool)


    # Start training
    start = datetime.now()
    for iter in range(args.num_iterations):
        #### Update potential ####
        for p in netD.parameters():
            p.requires_grad = True

        for _ in range(args.K_v):                         
            netD.zero_grad()

            # Sample x, t
            with torch.no_grad():
                x0 = source_dataset.sample().to(device)
                t = 0.9 * torch.rand(batch_size, device=device) + 0.05
                _, x0_latent = editor.run_on_batch(x0)
                diff_latent = netG1(t, x0_latent)
                xt_latent = x0_latent + t[:,None,None] * diff_latent
                # xt = editor._latents_to_image(latent)
                
            Vx = netD(t, xt_latent)
            cost = args.tau / t * torch.sum((xt_latent - x0_latent).reshape(batch_size, -1)**2, dim=1)
            errDx = phi1(Vx - cost).mean()
            
            errDx.backward()


            with torch.no_grad():
                y1 = target_dataset.sample().to(device)
                t = 0.9 * torch.rand(batch_size, device=device) + 0.05
                _, y1_latent = editor.run_on_batch(y1)
                diff_latent = netG2(t, y1_latent)
                yt_latent = y1_latent + (1 - t[:,None,None])*diff_latent
            
            cost = args.tau / (1 - t) * torch.sum((y1_latent - yt_latent).reshape(batch_size, -1)**2, dim=1)
            Vy = netD(t, yt_latent)
            errDy = phi2(- Vy - cost).mean()
            errDy.backward()
            optimizerD.step()


        #### Update Generator ####
        for p in netD.parameters():
            p.requires_grad = False

        for _ in range(args.K_T):
            netD.zero_grad()
            netG1.zero_grad()

            x0 = source_dataset.sample().to(device)
            t = 0.9 * torch.rand(batch_size, device=device) + 0.05
            _, x0_latent = editor.run_on_batch(x0)
            diff_latent = netG1(t, x0_latent)
            xt_latent = x0_latent + t[:,None,None] * diff_latent
            
            Vx = netD(t, xt_latent)
            cost = args.tau / t * torch.sum((xt_latent - x0_latent).reshape(batch_size, -1)**2, dim=1)
            errG1 = cost.mean() - Vx.mean()
            errG1.backward()
            optimizerG1.step()

            netD.zero_grad()
            netG2.zero_grad()

            t = 0.9 * torch.rand(batch_size, device=device) + 0.05
            y1 = target_dataset.sample().to(device)
            _, y1_latent = editor.run_on_batch(y1)
            diff_latent = netG2(t, y1_latent)
            yt_latent = y1_latent + (1 - t[:,None,None])*diff_latent
            
            Vy = netD(t, yt_latent)
            cost = args.tau / (1 - t) * torch.sum((y1_latent - yt_latent).reshape(batch_size, -1)**2, dim=1)
            errG2 = cost.mean() + Vy.mean()
            errG2.backward()
            optimizerG2.step()
        
        #### Update Schedulers
        if args.lr_scheduler:
            schedulerG1.step()
            schedulerG2.step()
            schedulerD.step()   

        log = f'Iteration {iter + 1:07d} : G1 Loss {errG1.item():.4f}, G2 Loss {errG2.item():.4f}, Dx Loss {errDx.item():.4f}, Dy Loss {errDy.item():.4f}, Elapsed {datetime.now() - start}'
        logger(log)     
        info = {'netE1': netG1, 'netE2': netG2, 'netD': netD, 'optimizerE1': optimizerG1, 'optimizerE2': optimizerG2}
        t = 0.5 * torch.ones(batch_size)

        with torch.no_grad():
            logger.save_image(info, t)
            logger.save_ckpt(info)
        logger.step()



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Bidirectional OT Parameters')

    # Experiment description
    parser.add_argument('--seed', type=int, default=1024, help='seed used for initialization')
    parser.add_argument('--exp', default='temp', help='name of the experiment')
    parser.add_argument('--problem_name', default='ffhq-young_to_ffhq-old', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=256, help='size of image (or data)')
    parser.add_argument('--num_channels', type=int, default=3, help='channel of image')
    
    # Training/Optimizer configurations
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
    parser.add_argument('--num_iterations', type=int, default=5000, help='the number of iterations')
    parser.add_argument('--lr_g', type=float, default=1.0e-5, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1.0e-5, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam')
    parser.add_argument('--lr_scheduler', action='store_true', default=False, help='Use lr scheduler or not. We use cosine scheduler if the argument is activated.')
    parser.add_argument('--eta_min', type=float, default=1e-5, help='eta_min of lr_scheduler')
    parser.add_argument('--K_v', type=int, default=1)
    parser.add_argument('--K_T', type=int, default=1)
    parser.add_argument('--use_ema', action='store_true', default=False)
    
    # (ADD) Important Hyperparameters    
    parser.add_argument('--lmbda', type=float, default=0, help='regularization parameter (HJB)')
    parser.add_argument('--tau', type=float, default=0.001, help='cost coefficient')
    parser.add_argument('--phi1', type=str, default='linear', choices=['linear', 'kl', 'chi', 'softplus'])
    parser.add_argument('--phi2', type=str, default='linear', choices=['linear', 'kl', 'chi', 'softplus'])

    # Visualize/Save configurations
    parser.add_argument('--print_every', type=int, default=10, help='print current loss for every x iterations')
    parser.add_argument('--save_ckpt_every', type=int, default=10000, help='save ckpt every x epochs')
    parser.add_argument('--save_image_every', type=int, default=1000, help='save images every x epochs')
    parser.add_argument('--fid_every', type=int, default=0)
    args = parser.parse_args()

    args.size = [3, args.image_size, args.image_size]

    train(args)
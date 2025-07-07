import torch
from torch.utils.data import DataLoader
import random

class Sampler:
    def __init__(
        self, device='cpu',
    ):
        self.device = device
    
    def sample(self, size=5):
        pass
    
    
    
class DatasetSampler(Sampler):
    def __init__(self, dataset, flag_label, batch_size, num_workers=40, device='cpu'):
        super(DatasetSampler, self).__init__(device=device)
        
        self.loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        self.flag_label = flag_label
        
        with torch.no_grad():
            self.dataset = torch.cat(
                [X for (X, y) in self.loader]
                ) if self.flag_label else torch.cat(
                [X for X in self.loader])
 
                
        
    def sample(self, batch_size=8):
        ind = random.choices(range(len(self.dataset)), k=batch_size)
        with torch.no_grad():
            batch = self.dataset[ind].clone().to(self.device).float()
        return batch


import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import os.path
import torch.distributions as td
from functools import partial
import torchvision


class ToySampler: # a dump prior sampler to align with DataSampler
    def __init__(self, dataset):
        self.dataset = dataset

    def sample(self, batch_size):
        return self.dataset.sample(batch_size)

# ------------------------
# Toy Datasets
# ------------------------
def get_toydataset(data_name, datasize):
    return {'1d_2gaussian': MixMultiVariateNormal1D,
    '8gaussian': MixMultiVariateNormal,
    'checkerboard': CheckerBoard,
    'spiral': Spiral,
    'moon': Moon,
    '25gaussian': SquareGaussian,
    'twocircles': partial(Circles, centers=[[0,0], [0,0]], radius=[8,16], sigmas=[0.2, 0.2]),
    }.get(data_name)(datasize)


class CheckerBoard:
    def __init__(self, datasize):
        pass

    def sample(self, n):
        n = n[0]
        n_points = 3*n
        n_classes = 2
        freq = 5
        x = np.random.uniform(-(freq//2)*np.pi, (freq//2)*np.pi, size=(n_points, n_classes))
        mask = np.logical_or(np.logical_and(np.sin(x[:,0]) > 0.0, np.sin(x[:,1]) > 0.0), \
        np.logical_and(np.sin(x[:,0]) < 0.0, np.sin(x[:,1]) < 0.0))
        y = np.eye(n_classes)[1*mask]
        x0=x[:,0]*y[:,0]
        x1=x[:,1]*y[:,0]
        sample=np.concatenate([x0[...,None],x1[...,None]],axis=-1)
        sqr=np.sum(np.square(sample),axis=-1)
        idxs=np.where(sqr==0)
        sample=np.delete(sample,idxs,axis=0)
        sample=torch.Tensor(sample)
        sample=sample[0:n,:]
        return sample / 3.


class Spiral:
    def __init__(self, datasize):
        pass

    def sample(self, n):
        n = n[0]
        theta = np.sqrt(np.random.rand(n))*3*np.pi-0.5*np.pi # np.linspace(0,2*pi,100)

        r_a = theta + np.pi
        data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
        x_a = data_a + 0.25*np.random.randn(n,2)
        samples = np.append(x_a, np.zeros((n,1)), axis=1)
        samples = samples[:,0:2]
        return torch.Tensor(samples)


class Moon:
    def __init__(self, datasize):
        pass

    def sample(self, n):
        n = n[0]
        x = np.linspace(0, np.pi, n // 2)
        u = np.stack([np.cos(x) + .5, -np.sin(x) + .2], axis=1) * 12.
        u += 0.5*np.random.normal(size=u.shape)
        v = np.stack([np.cos(x) - .5, np.sin(x) - .2], axis=1) * 12.
        v += 0.5*np.random.normal(size=v.shape)
        x = np.concatenate([u, v], axis=0)
        return torch.Tensor(x)


class MixMultiVariateNormal1D:
    def __init__(self, datasize ,sigma=0.1):
        self.mus = [-2, 2]
        self.sigma = sigma

    def sample(self, n):
        n = n[0]
        ind_sample = n / 2
        samples=[torch.randn(int(ind_sample),1)*self.sigma + mu for mu in self.mus]
        samples=torch.cat(samples,dim=0)
        return samples


class MixMultiVariateNormal:
    def __init__(self, datasize, radius=12, num=8, sigma=0.4):

        # build mu's and sigma's
        arc = 2*np.pi/num
        xs = [np.cos(arc*idx)*radius for idx in range(num)]
        ys = [np.sin(arc*idx)*radius for idx in range(num)]
        mus = [torch.Tensor([x,y]) for x,y in zip(xs,ys)]
        dim = len(mus[0])
        sigmas = [sigma*torch.eye(dim) for _ in range(num)] 

        self.num = num
        self.dists=[
            td.multivariate_normal.MultivariateNormal(mu, sigma) for mu, sigma in zip(mus, sigmas)
        ]

    def sample(self, n):
        n = n[0]
        assert n % self.num == 0
        ind_sample = n/self.num
        samples=[dist.sample([int(ind_sample)]) for dist in self.dists]
        samples=torch.cat(samples,dim=0)
        return samples


class SquareGaussian:
    def __init__(self, datasize, num=25, sigma=0.01):

        # build mu's and sigma's
        xs = [-16]*5+[-8]*5+[0]*5+[8]*5+[16]*5
        ys = [-16,-8,0,8,16]*5
        mus = [torch.Tensor([x,y]) for x,y in zip(xs,ys)]
        dim = len(mus[0])
        sigmas = [sigma*torch.eye(dim) for _ in range(num)] 

        self.num = num
        self.dists=[
            td.multivariate_normal.MultivariateNormal(mu, sigma) for mu, sigma in zip(mus, sigmas)
        ]

    def sample(self, n):
        n = n[0]
        assert n%self.num == 0
        ind_sample = n/self.num
        samples=[dist.sample([int(ind_sample)]) for dist in self.dists]
        samples=torch.cat(samples,dim=0)
        return samples
    

class Circles:
    def __init__(self, datasize, centers, radius, sigmas):
        assert len(centers) == len(radius)
        assert  len(radius) == len(sigmas)
        self.num_circles = len(centers)        
        self.centers = centers
        self.radius = radius
        self.sigmas = sigmas
        
    def sample(self, n):
        n = n[0]
        assert n % self.num_circles == 0
        ind_sample =  int(n // self.num_circles)
        centers = torch.tensor(self.centers * ind_sample, dtype=torch.float32)
        radius = torch.tensor(self.radius * ind_sample, dtype=torch.float32)[:,None]
        sigmas = torch.tensor(self.sigmas * ind_sample, dtype=torch.float32)[:,None]
        noise = torch.randn(size=(n, 2))
        z = torch.randn(size=(n, 2))
        z = z/torch.norm(z, dim=1, keepdim=True)
        return centers + radius* z + sigmas * noise



class LoaderSampler(Sampler):
    def __init__(self, loader, device='cpu'):
        super(LoaderSampler, self).__init__(device)
        self.loader = loader
        self.it = iter(self.loader)
        
    def sample(self, size=5):
        assert size <= self.loader.batch_size
        try:
            batch, _ = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(batch) < size:
            return self.sample(size)
            
        return batch[:size].to(self.device)
    



def load_dataset(name, path, img_size=64, batch_size=64, shuffle=True, device='cuda'):
    # In case of using certain classe from the MNIST dataset you need to specify them by writing in the next format "MNIST_{digit}_{digit}_..._{digit}"
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Lambda(lambda x: 2 * x - 1)
    ])

    dataset_name = name.split("_")[0]
    is_colored = False

    classes = [int(number) for number in name.split("_")[1:]]
     
    if not classes:
        classes = [i for i in range(10)]

    train_set = torchvision.datasets.MNIST(path, train=True, transform=transform, download=True)
    test_set = torchvision.datasets.MNIST(path, train=False, transform=transform, download=True)

    train_test = []

    for dataset,lbl in zip([train_set, test_set],["train","test"]):
        data = []
        labels = []
        for k in range(len(classes)):
            data.append(torch.stack(
                [dataset[i][0] for i in range(len(dataset.targets)) if dataset.targets[i] == classes[k]],
                dim=0
            ))
            labels += [k]*data[-1].shape[0]
             
            
        data = torch.cat(data, dim=0)
        data = data.reshape(-1, 1, 32, 32)
        labels = torch.tensor(labels)
        

        # if is_colored:
        #     data = get_random_colored_images(data)
        
        if lbl == "train":
            train_test.append(TensorDataset(data[:12_211], labels[:12_211]))
        else:
            train_test.append(TensorDataset(data[:2_063], labels[:2_063]))

    train_set, test_set = train_test
    
    #train_set  = train_set[:12_211]  
    #test_set = test_set[:2_063]
    
     
    train_sampler = LoaderSampler(DataLoader(train_set, shuffle=shuffle, num_workers=8, batch_size=batch_size), device)
    test_sampler = LoaderSampler(DataLoader(test_set, shuffle=shuffle, num_workers=8, batch_size=batch_size), device)
    return train_set, test_set, train_sampler, test_sampler
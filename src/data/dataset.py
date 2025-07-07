import os
import random
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
import io
import os.path
import pickle
from PIL import Image
import torch.distributions as td
from functools import partial
import random


ROOT_PATH = "./data"
# ROOT_PATH = "./"


def build_boundary_distribution(args):
    problem_name = args.problem_name
    train = args.train # (train=True, test=False)
    datasize = args.size # (c, h, w)

    if problem_name.find('_to_') != -1:
        source_data_name, target_data_name = problem_name.split('_to_')
    else:
        source_data_name = 'gaussian'
        target_data_name = problem_name
    
    source_sampler = build_sampler(source_data_name, train, datasize, args.batch_size)
    target_sampler = build_sampler(target_data_name, train, datasize, args.batch_size)

    return source_sampler, target_sampler


def build_sampler(data_name, train, datasize, batch_size):    
    if data_name == 'gaussian':
        dataset = td.MultivariateNormal(torch.zeros(datasize), 1*torch.eye(datasize[-1]))
        sampler = ToySampler(dataset, batch_size)
    
    elif data_name == 'uniform':
        dataset = td.Uniform(torch.zeros(datasize)-1, torch.ones(datasize))
        sampler = ToySampler(dataset, batch_size)

    elif len(datasize) == 1:
        try:
            dataset = get_toydataset(data_name, datasize)
            sampler = ToySampler(dataset, batch_size)
        except:
            if 'gaussian' in data_name:
                a = float(data_name.split('n')[1])
                dataset = td.Normal(a * torch.ones(datasize,), 1)
                sampler = ToySampler(dataset, batch_size)
            else: NotImplementedError
        
    else:
        dataloader = get_dataloader(data_name, train, datasize, batch_size)
        sampler = Sampler(dataloader)

    return sampler
    

class Sampler: # a dump data sampler
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def sample(self):
        try: 
            data = next(self.iterloader)
        except:
            self.iterloader = iter(self.dataloader)
            data = next(self.iterloader)
        
        try: data , _ = data
        except: pass
        
        return data.float()


class ToySampler: # a dump prior sampler to align with DataSampler
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def sample(self):
        return self.dataset.sample([self.batch_size])


def get_dataloader(data_name, train, datasize, batch_size, drop_last=True):
    num_workers = 4
    if data_name == 'mnist':
        dataset = MNIST(ROOT_PATH, train=train, transform=transforms.Compose([
                        transforms.Resize(datasize[1]),
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)
    
    elif data_name == 'cmnist':
        class CMNIST(Dataset):
            def __init__(self, root, train):
                self.dataset = MNIST(root, train=train, transform=transforms.Compose([
                        transforms.Resize(datasize[1]),
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),]), download=True)

                self.rgb = [torch.tensor([1,0,0]).float()[:,None,None], 
                            torch.tensor([0,1,0]).float()[:,None,None], 
                            torch.tensor([0,0,1]).float()[:,None,None]]

            def __getitem__(self, index):
                x, _ = self.dataset[index]
                # make into rgb randomly
                p = int(3 * random.random())
                x = self.rgb[p] * x
                
                return 2 * x - 1

            def __len__(self):
                return self.dataset.__len__()

        dataset = CMNIST(ROOT_PATH, train)

    elif data_name == 'cifar10':
        dataset = CIFAR10(ROOT_PATH, train=True, transform=transforms.Compose([
                        transforms.Resize(datasize[-2:]),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)
    
    elif data_name == 'celeba':
        num_workers = 0
        train_transform = transforms.Compose([
            transforms.CenterCrop(140),
            transforms.Resize(datasize[-2:]),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = CelebA64(
            root = os.path.join(ROOT_PATH, 'celeba/celeba_lmdb'),
            transform=train_transform
        )

    elif data_name == 'celeba_hq':
        train_transform = transforms.Compose([
                transforms.Resize(datasize[-2:]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        dataset = CelebA_HQ(
            root=os.path.join(ROOT_PATH, 'celeba-hq/celeba-256'),
            partition_path=os.path.join(ROOT_PATH,'celeba-hq/list_eval_partition_celeba.txt'),
            mode='train', # 'train', 'val', 'test'
            transform=train_transform,
        )
    
    # Now it is I2I task
    else:
        ##### get data path #####
        I2I_FOLDERS = [folder for folder in os.listdir(ROOT_PATH) 
                       if folder.find('2') != -1 and folder.find('.') == -1]
        folder_name = [folder for folder in I2I_FOLDERS if folder.find(data_name) != -1]
        if len(folder_name) > 1 or len(folder_name) == 0: 
            print(f'[WARNING] Names are duplicated: {folder_name[0]}, {folder_name[1]}, ...')
        
        trainortest = 'train' if train else 'test'
        AorB = 'A' if folder_name[0].find(data_name) == 0 else 'B'

        data_path = os.path.join(ROOT_PATH, folder_name[0], f'{trainortest}{AorB}')


        ##### get dataset #####
        # if data_name in ['male', 'female']:
        #     train_transform = transforms.Compose([
        #             transforms.CenterCrop(140),
        #             transforms.Resize(datasize[-2:]),
        #             transforms.RandomHorizontalFlip(),
        #             transforms.ToTensor(),
        #             transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        #         ])
        # else:
        train_transform = transforms.Compose([
                transforms.Resize(datasize[-2:]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])

        dataset = UnalignDataset(data_path, train_transform)

    # shuffle = True if train else False
    shuffle = True
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    return data_loader


class CelebA64(data.Dataset):
    '''Note: CelebA (Total 202599 iamges) in 3x64x64 dim'''
    def __init__(self, root, transform):
        super().__init__()
        self.root = root
        self.transform = transform

        import lmdb
        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

        cache_file = os.path.join(self.root, '_cache_')
        
        # av end
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor()]
            pickle.dump(self.keys, open(cache_file, "wb"))

    def __getitem__(self, index):
        img, target = None, -1
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target
    
    def __len__(self):
        return self.length


class CelebA_HQ(data.Dataset):
    '''Note: CelebA (about 200000 images) vs CelebA-HQ (30000 images)'''
    def __init__(self, root, partition_path, mode='train', transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform

        # Split train/val/test 
        self.partition_dict = {}
        self.get_partition_label(partition_path)
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        self.save_img_path()
        print('[Celeba-HQ Dataset]')
        print(f'Train {len(self.train_dataset)} | Val {len(self.val_dataset)} | Test {len(self.test_dataset)}')

        if mode == 'train':
            self.dataset = self.train_dataset
        elif mode == 'val':
            self.dataset = self.val_dataset
        elif mode == 'test':
            self.dataset = self.test_dataset
        else:
            raise ValueError

    def get_partition_label(self, list_eval_partition_celeba_path):
        '''Get partition labels (Train 0, Valid 1, Test 2) from CelebA
        See "celeba/Eval/list_eval_partition.txt"
        '''
        with open(list_eval_partition_celeba_path, 'r') as f:
            for line in f.readlines():
                filenum = line.split(' ')[0].split('.')[0] # Use 6-digit 'str' instead of int type
                partition_label = int(line.split(' ')[1]) # 0 (train), 1 (val), 2 (test)
                self.partition_dict[filenum] = partition_label

    def save_img_path(self):
        for filename in os.listdir(self.root):
            assert os.path.isfile(os.path.join(self.root, filename))
            filenum = filename.split('.')[0]
            label = self.partition_dict[filenum]
            if label == 0:
                self.train_dataset.append(os.path.join(self.root, filename))
            elif label == 1:
                self.val_dataset.append(os.path.join(self.root, filename))
            elif label == 2:
                self.test_dataset.append(os.path.join(self.root, filename))
            else:
                raise ValueError

    def __getitem__(self, index):
        img_path = self.dataset[index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.dataset)


class UnalignDataset(data.Dataset):
    def __init__(self, data_path, transform):
        self.data_path = data_path
        self.paths = sorted(make_dataset(data_path))
        self.transform = transform

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img)

    def __len__(self):
        return len(self.paths)


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG'])


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

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
        u = np.stack([np.cos(x) + .5, -np.sin(x) + .2], axis=1) * 10.
        u += 0.5*np.random.normal(size=u.shape)
        v = np.stack([np.cos(x) - .5, np.sin(x) - .2], axis=1) * 10.
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

if __name__ == '__main__':
    import argparse
    from torchvision.utils import save_image
    parser = argparse.ArgumentParser('UOTM parameters')
    
    # Experiment description
    parser.add_argument('--problem_name', type=str)
    parser.add_argument('--size', nargs='+', type=int, help='size of image (or data)')
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    priorsampler, datasampler = build_boundary_distribution(args)

    for i in range(200):
        x = datasampler.sample()
        save_image(0.5*x+0.5, 'hi.jpg')
        x = priorsampler.sample()
        save_image(0.5*x+0.5, 'hi2.jpg')
        print(x.shape)
        print('-------------------------')
        break

    print('Succesfully sampled')

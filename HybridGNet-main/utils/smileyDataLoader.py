import os 

import torch
import pathlib
import re

from skimage import io, transform

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2 
from .smileyGraphDataLoader import getSeg, getHeart

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


class LandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_path, label_path, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_path = img_path
        self.label_path = label_path
        self.transform = transform
                
        data_root = pathlib.Path(img_path)
        all_files = list(data_root.glob('*.png'))
        all_files = [str(path) for path in all_files]
        all_files.sort(key = natural_key)
        
        self.images = all_files
        print('Total of landmarks:', len(all_files))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]
        image = io.imread(img_name, as_gray= True).astype('float')#/255
        # image = io.imread(img_name).astype('float').astype('float') /255
        # Normalize to [0, 1]
        #image = (image - image.min()) / (image.max() - image.min())/255

        image = np.expand_dims(image, axis=2)
        
        label = img_name.replace(os.path.normpath(self.img_path), os.path.normpath(self.label_path)).replace('.png', '.npy')

        landmarks = np.load(label)
        landmarks = landmarks.astype('float').reshape(-1, 2) # shape and stuff does not change here but makes it float
        # print("print second", image.shape, landmarks.shape)
        
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

    
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple, list))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        # print(image.shape, [new_h, new_w])
        img = transform.resize(image, (new_h, new_w))
        # print("after transform2", img.shape, [new_h, new_w])

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomScale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']       
        
        # Pongo limites para evitar que los landmarks salgan del contorno
        min_x = np.min(landmarks[:,0]) 
        max_x = np.max(landmarks[:,0])
        ancho = max_x - min_x
        
        min_y = np.min(landmarks[:,1])
        max_y = np.max(landmarks[:,1])
        alto = max_y - min_y
        
        max_var_x = 320 / ancho 
        max_var_y = 256 / alto
                
        min_var_x = 0.80 # dunno what these are
        min_var_y = 0.80
                                
        varx = np.random.uniform(min_var_x, max_var_x)
        vary = np.random.uniform(min_var_x, max_var_y)
                
        landmarks[:,0] = landmarks[:,0] * varx
        landmarks[:,1] = landmarks[:,1] * vary
        
        h, w = image.shape[:2]
        new_h = np.round(h * vary).astype('int')
        new_w = np.round(w * varx).astype('int')

        # print(image.shape, [new_h, new_w]) 
        img = transform.resize(image, (new_h, new_w))
        # print("after transform", img.shape, [new_h, new_w])
        
        # Cropeo o padeo aleatoriamente
        min_x = np.round(np.min(landmarks[:,0])).astype('int')
        max_x = np.round(np.max(landmarks[:,0])).astype('int')
        
        min_y = np.round(np.min(landmarks[:,1])).astype('int')
        max_y = np.round(np.max(landmarks[:,1])).astype('int')
        # print("minx max x y y " , min_x, max_x, min_y, max_y)
        
        if new_h > 320:
            rango = 320 - (max_y - min_y)
            maxl0y = new_h - 321
            
            if rango > 0 and min_y > 0:
                l0y = min_y - np.random.randint(0, min(rango, min_y))
                l0y = min(maxl0y, l0y)
            else:
                l0y = min_y
                
            l1y = l0y + 320
            
            img = img[l0y:l1y,:]
            landmarks[:,1] -= l0y
            
        elif new_h < 320:
            pad = h - new_h
            p0 = np.random.randint(np.floor(pad/4), np.ceil(3*pad/4))
            p1 = pad - p0
            
            img = np.pad(img, ((p0, p1), (0, 0), (0, 0)), mode='constant', constant_values=0)
            landmarks[:,1] += p0
        
        if new_w > 256:
            rango = 256 - (max_x - min_x)
            maxl0x = new_w - 257
            
            if rango > 0 and min_x > 0:
                l0x = min_x - np.random.randint(0, min(rango, min_x))
                l0x = min(maxl0x, l0x)
            else:
                l0x = min_x
            
            l1x = l0x + 256
                
            img = img[:, l0x:l1x]
            landmarks[:,0] -= l0x
            
        elif new_w < 256:
            pad = w - new_w
            p0 = np.random.randint(np.floor(pad/4), np.ceil(3*pad/4))
            p1 = pad - p0
            
            img = np.pad(img, ((0, 0), (p0, p1), (0, 0)), mode='constant', constant_values=0)
            landmarks[:,0] += p0
        
        if img.shape[0] != 320 or img.shape[1] != 256:
            print('Original', [new_h,new_w])
            print('Salida', img.shape)
            raise Exception('Error')
            
        return {'image': img, 'landmarks': landmarks}
    
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return np.float32(cv2.LUT(image.astype('uint8'), table))

class AugColor(object):
    def __init__(self, gammaFactor):
        self.gammaf = gammaFactor

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        # Gamma
        gamma = np.random.uniform(1 - self.gammaf, 1 + self.gammaf / 2)
        
        image[:,:,0] = adjust_gamma(image[:,:,0] * 255, gamma) / 255
        
        # Adds a little noise
        image = image + np.random.normal(0, 1/128, image.shape)
        #image = image + np.random.normal(0, 1/512, image.shape)
        
        return {'image': image, 'landmarks': landmarks}

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        angle = np.random.uniform(- self.angle, self.angle)

        image = transform.rotate(image, angle)
        
        centro = image.shape[0] / 2, image.shape[1] / 2
        
        landmarks -= centro
        
        theta = np.deg2rad(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        
        landmarks = np.dot(landmarks, R)
        
        landmarks += centro

        return {'image': image, 'landmarks': landmarks}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
                
        size = image.shape[0]
        image = image.transpose((2, 0, 1))
        landmarks = landmarks.reshape(-1, 2) / size
        landmarks = np.clip(landmarks, 0, 1)
        
        return {'image': torch.from_numpy(image).float(),
                'landmarks': torch.from_numpy(landmarks).float()}
    
class ToTensorSeg(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
                
        size = image.shape[0]
        image = image.transpose((2, 0, 1))
        landmarks = landmarks.reshape(-1, 2) / size
        landmarks = np.clip(landmarks, 0, 1)
        seg = getSeg(landmarks * [320,256]) # for the unet
        
        return {'image': torch.from_numpy(image).float(),
                'landmarks': torch.from_numpy(landmarks).float(),
                'seg': torch.from_numpy(seg).long()}
    
class ToTensorLungs(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
                
        size = image.shape[0]
        image = image.transpose((2, 0, 1))
        landmarks = landmarks.reshape(-1, 2) / size
        landmarks = np.clip(landmarks, 0, 1)[:2]
        
        return {'image': torch.from_numpy(image).float(),
                'landmarks': torch.from_numpy(landmarks).float()}
    
class ToTensorLH(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
                
        size = image.shape
        image = image.transpose((2, 0, 1))
        landmarks = landmarks.reshape(-1, 2) / np.array(size[:2])
        landmarks = np.clip(landmarks, 0, 1)[:14]
        # print("to tensor LH", landmarks.shape)
        return {'image': torch.from_numpy(image).float(),
                'landmarks': torch.from_numpy(landmarks).float()}
    
class ToTensorSegLH(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
                
        size = image.shape
        image = image.transpose((2, 0, 1))
        landmarks = landmarks.reshape(-1, 2) / np.array(size[:2])
        landmarks = np.clip(landmarks, 0, 1)[:14]
        seg = getHeart(landmarks * [320,256]) # for the unet
        
        return {'image': torch.from_numpy(image).float(),
                'landmarks': torch.from_numpy(landmarks).float(),
                'seg': torch.from_numpy(seg).long()}

class ToTensorSeg1(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
                 
        size = image.shape[0]
        image = image.transpose((2, 0, 1))
        landmarks = landmarks / size
        landmarks = np.clip(landmarks, 0, 1) * 320

        RL = landmarks[:1,:]
        LL = landmarks[2:12,:]

        if landmarks.shape[0] == 2:
            seg = getDenseMask(RL, LL)
        elif landmarks.shape[0] == 14:
            H = landmarks[2:14,:]
            seg = getDenseMask(RL, LL, H)
        return {'image': torch.from_numpy(image).float(),
                'seg': torch.from_numpy(seg).long()}
    

def getDenseMask(RL, LL, H = None, CLA1 = None, CLA2 = None):
    img = np.zeros([320,256])
    
    RL = RL.reshape(-1, 1, 2).astype('int')
    LL = LL.reshape(-1, 1, 2).astype('int')

    img = cv2.drawContours(img, [RL], -1, 1, -1)
    img = cv2.drawContours(img, [LL], -1, 1, -1)
    
    if H is not None:
        H = H.reshape(-1, 1, 2).astype('int')
        img = cv2.drawContours(img, [H], -1, 2, -1)
        
    if CLA1 is not None:
        CLA1 = CLA1.reshape(-1, 1, 2).astype('int')
        img = cv2.drawContours(img, [CLA1], -1, 3, -1)
    
    if CLA2 is not None:
        CLA2 = CLA2.reshape(-1, 1, 2).astype('int')
        img = cv2.drawContours(img, [CLA2], -1, 3, -1)
    
    return img
import os
import torch
import torch.nn.functional as F
import argparse

from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import scipy.sparse as sp
import numpy as np

from sklearn.metrics import mean_squared_error
from utils.utilsSmiley import scipy_to_torch_sparse, genMatrixes, genMatrixesLH
from utils.smileyDataLoader import LandmarksDataset, ToTensor, ToTensorLH, Rescale, RandomScale, AugColor, Rotate

from models.hybridDoubleSkip import Hybrid as DoubleSkip
from models.hybridSkip import Hybrid as Skip
from models.hybrid import Hybrid
from models.hybridNoPool import Hybrid as HybridNoPool

from models.pca import PCA_Net
from models.vae import VAE_Mixed

from skimage.metrics import hausdorff_distance as hd

def hd_land(target, pred, shape):
    set_ax = target[:,0].tolist()
    set_ay = target[:,1].tolist()

    set_bx = pred[:,0].tolist() 
    set_by = pred[:,1].tolist() 

    coords_a = np.zeros(shape, dtype=bool)
    coords_b = np.zeros(shape, dtype=bool)
    for x, y in zip(set_ax, set_ay):
        coords_a[(x, y)] = True

    for x, y in zip(set_bx, set_by):
        coords_b[(x, y)] = True
    
    dist = hd(coords_a, coords_b)
    
    return dist
    
def hd_landmarks(out, label, size = 200):
    shape = (size, size)

    target = np.round(label.cpu().numpy()*size).astype('int32').clip(0, size - 1)
    pred = np.round(out.cpu().numpy()*size).astype('int32').clip(0, size - 1)
    
    d_lungs = hd_land(target[:2,:], pred[:2,:], shape)
    d_heart = hd_land(target[2:14,:], pred[2:14,:], shape)
    
    if target.shape[0] > 14:
        d_cla =  hd_land(target[14:,:], pred[14:,:], shape)
        return (d_lungs + d_heart + d_cla) / 3
    else:
        return (d_lungs + d_heart) / 2



def trainer(train_dataset, val_dataset, model, config):
    torch.manual_seed(420)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = model.to(device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True, num_workers = 0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = config['val_batch_size'], num_workers = 0)

    optimizer = torch.optim.Adam(params = model.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])

    train_loss_avg = []
    val_loss_avg = []
    val_hd_avg = []

    tensorboard = "Training"
    print("config['name']", config['name'])
        
    folder = os.path.join(tensorboard, config['name'])

    try:
        os.mkdir(folder)
    except:
        pass 

    writer = SummaryWriter(log_dir = folder)  

    best = 1e12
    bestHD = 1e12
    bestMSE = 1e12
    suffix = ".pt"
    
    print('Training ...')
        
    scheduler = StepLR(optimizer, step_size=config['stepsize'], gamma=config['gamma'])
    
    for epoch in range(config['epochs']):
        model.train()

        train_loss_avg.append(0)
        num_batches = 0
        print("first element in trainloader", train_loader)
        for sample_batched in train_loader:
            image, target = sample_batched['image'].to(device), sample_batched['landmarks'].to(device)
            
            print("image", image.shape)
            print("target", target.shape)
            out = model(image)
            out = out.reshape(out.shape[0], -1, 2)
            print("out", out.shape)
            
            optimizer.zero_grad()
            
            loss = F.mse_loss(out, target)
            train_loss_avg[-1] += loss.item()
            
            if config['model'] == 'VAE':
                loss += 1e-5 * (-0.5 * torch.mean(torch.mean(1 + model.log_var - model.mu ** 2 - model.log_var.exp(), dim=1), dim=0))
                
            loss.backward()

            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()

            num_batches += 1

        train_loss_avg[-1] /= num_batches

        print('Epoch [%d / %d] train average reconstruction error: %f' % (epoch+1, config['epochs'], train_loss_avg[-1]*200*200))

        num_batches = 0

        model.eval()
        val_loss_avg.append(0)
        val_hd_avg.append(0)

        with torch.no_grad():
            for sample_batched in val_loader:
                image, target = sample_batched['image'].to(device), sample_batched['landmarks'].to(device)

                out = model(image)
                out = out.reshape(-1, 2)
                target = target.reshape(-1, 2)
                
                dist = hd_landmarks(out, target, config['inputsize'])
                val_hd_avg[-1] += dist 

                loss_rec = mean_squared_error(out.cpu().numpy(), target.cpu().numpy())
                val_loss_avg[-1] += loss_rec
                num_batches += 1   
                loss_rec = 0

        val_loss_avg[-1] /= num_batches
        val_hd_avg[-1] /= num_batches
        
        print('Epoch [%d / %d] validation average reconstruction error: %f' % (epoch+1, config['epochs'], val_loss_avg[-1] * 200 * 200))

        writer.add_scalar('Train/Loss', train_loss_avg[-1], epoch)
        writer.add_scalar('Train/MSE', train_loss_avg[-1] * 200 * 200, epoch)
        
        writer.add_scalar('Validation/MSE', val_loss_avg[-1]  * 200 * 200, epoch)
        writer.add_scalar('Validation/Hausdorff Distance', val_hd_avg[-1], epoch)
        
        if epoch % 200 == 0:
            suffix = "_%s.pt" % epoch
            best = 1e12
            bestHD = 1e12
            
        if val_loss_avg[-1] < best:
            best = val_loss_avg[-1]
            print('Model Saved MSE')
            out = "bestMSE.pt"
            torch.save(model.state_dict(), os.path.join(folder, out.replace('.pt',suffix)))

        if val_loss_avg[-1] < bestMSE:
            bestMSE = val_loss_avg[-1]
            print('Model Saved MSE all time')
            torch.save(model.state_dict(), os.path.join(folder, "bestMSE.pt"))

        if val_hd_avg[-1] < bestHD:
            bestHD = val_hd_avg[-1]
            print('Model Saved HD')
            out = "bestHD.pt"
            torch.save(model.state_dict(), os.path.join(folder, out.replace('.pt',suffix)))

        scheduler.step()
    
    torch.save(model.state_dict(), os.path.join(folder, "final.pt"))

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", type=str)    
    parser.add_argument("--load", help="enter the folder where the weights are saved", default = "None", type=str)
    parser.add_argument("--inputsize", default = [320,256], type=int)
    parser.add_argument("--epochs", default = 2500, type = int)
    parser.add_argument("--lr", default = 1e-4, type = float)
    parser.add_argument("--stepsize", default = 50, type = int)
    parser.add_argument("--gamma", default = 0.9, type = float)
    parser.add_argument("--model", default = 'PCA', type=str)    
    
    # Define the output: All organs (including clavicles), Lungs and Heart, only Lungs TBD
    parser.add_argument('--allOrgans', dest='allOrgans', action='store_true')
    parser.set_defaults(allOrgans=False)
        
    parser.add_argument('--extended', dest='extended', action='store_true')
    parser.set_defaults(extended=False)
    
    config = parser.parse_args()
    config = vars(config)

    if config['allOrgans']:
        print('Organs: Lungs Heart and Clavicles')
        A, AD, D, U = genMatrixes()
        ToTensor = ToTensor
    else:
        print('Organs: Lungs and Heart')
        A, AD, D, U = genMatrixesLH()
        print("A", A.shape)
        ToTensor = ToTensorLH
    
    inputSize = config['inputsize']
        
    if config['extended']:
        print('Extended dataset')
        # train_path = "Datasets/Extended/Train"
        # val_path = "Datasets/Extended/Val" 
    else:
        print("else statement")
        # train_path = "Datasets/JSRT/Train"
        # val_path = "Datasets/JSRT/Val" 
        train_path ="../Speciale/generateData/data/train"
        val_path = "../Speciale/generateData/data/val"
        
    img_path = os.path.join(train_path, 'Images')
    label_path = os.path.join(train_path, 'landmarks')

    print('Training on:', train_path)
    train_dataset = LandmarksDataset(img_path=img_path,
                                     label_path=label_path,
                                     transform = transforms.Compose([
                                                 RandomScale(),
                                                 Rotate(3),
                                                 AugColor(0.40),
                                                 ToTensor()])
                                     )
    print(train_dataset.__len__() , 'Training samples')

    img_path = os.path.join(val_path, 'Images')
    label_path = os.path.join(val_path, 'landmarks')
    val_dataset = LandmarksDataset(img_path=img_path,
                                     label_path=label_path,
                                     transform = transforms.Compose([
                                                 Rescale(inputSize),
                                                 ToTensor()])
                                     )
 
    print(val_dataset.__len__() , 'Validation samples')
    config['latents'] = 64
    config['batch_size'] = 4
    config['val_batch_size'] = 1
    config['weight_decay'] = 1e-5
    config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config['name'] = "name"
    
    print("first config print here", config['model'])
    print("second config print here", config['name'])

    if config['model'] == "PCA": #the model you want default = "name"
        model = PCA_Net(config)
    elif config['model'] == "VAE":
        model = VAE_Mixed(config)
    else:
        print("hellooo")
        raise Exception('No valid model')
        
        
    trainer(train_dataset, val_dataset, model, config)
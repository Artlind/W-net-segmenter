"""
Training module
"""
import os
import torch
import torch.utils
import torch.utils.data
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
from Segmenter import Segmenter
from tqdm import tqdm
from skimage import io


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print('Initialisation of cuda')
    torch.cuda.init()
else:
    print('Mode CPU')
    DEVICE = torch.device('cpu')

# =============================================================================
# Losses functions
# =============================================================================

r = 5
sigmaI = 4.5
sigmaX = 3

def dist2c(i, j, device=DEVICE):
    return torch.tensor((i[0]-j[0])**2+(i[1]-j[1])**2.0).to(device)

def distvaluec(im, i, j, device=DEVICE):
    out = torch.tensor(0.0).to(device)
    for k in range(im.size()[0]):
        out += (im[k][i[0]][i[1]]-im[k][j[0]][j[1]])**2.0
    return out

def probclass(Uenc, imnumber, u, k):
    return Uenc[imnumber][k][u[0]][u[1]]

def weight(im, i, j, device=DEVICE):
    X = dist2c(i, j, device)
    if X >= r:
        return torch.tensor(0.0).to(device)
    f = distvaluec(im, i, j, device)
    return np.exp(-f/(sigmaI**2.0)-X/(sigmaX**2.0)).to(device)

def cover(resolution):
    list = []
    for i in range(resolution[0]):
        for j in range(resolution[1]):
            list.append((i, j))
    return list


def Ncutsoft(Uenc, batch_im, device=DEVICE):
    classes = Uenc.size()[1]
    batch = Uenc.size()[0]
    resolution = Uenc.size()[2:]
    N = torch.tensor(0.0).to(device)
    pixels = cover(resolution)
    for imnumber in tqdm(range(batch), desc="Images"):
        weights = {}
        S = torch.tensor(0.0).to(device)
        for k in range(classes):
            num = torch.tensor(0.0).to(device)
            den = torch.tensor(0.0).to(device)
            probsclass = {}
            for u in pixels:
                i, j = u
                ssnum = torch.tensor(0.0).to(device)
                ssden = torch.tensor(0.0).to(device)
                for l in range(max(0, i-r+1), min(i+r, resolution[0])):
                    for m in range(max(0, i-l+j-r+1), min(i-l+j+r, resolution[1])):
                        v = (l, m)
                        if v not in probsclass:
                            probsclass[v] = probclass(Uenc, imnumber, list(v), k)
                        if (u, v) not in weights:
                            weights[(u, v)] = weight(batch_im[imnumber], u, v, device)
                            weights[(v, u)] = weights[(u, v)]
                        ssnum += probsclass[v]*weights[(u, v)]
                        ssden += weights[(u, v)]
                if u not in probsclass:
                    probsclass[u] = probclass(Uenc, imnumber, list(u), k)
                num += probsclass[u]*ssnum
                den += probsclass[u]*ssden
            S += num/den
        N += classes-S
    return N/batch

def Jrecons(Udec, batch_im):
    batch = Udec.size()[0]
    return torch.norm(Udec-batch_im)**2/batch

# =============================================================================
# Training parameters
# =============================================================================



EPOCH = 10
BATCH = 2
LR = 0.003

def learn(data, K, resolution, epoch, device, checkpoint=None):
    """
    Data is a tensor ith size (len(dataset), number of colours,resolution[0], resolution[1])
    """
    model = Segmenter(resolution, K, device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    if checkpoint != None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    dataloader = torch.utils.data.DataLoader(data, BATCH,
                                             shuffle=True, drop_last=True)
    for i in tqdm(range(epoch), desc="Epochs"):
        for batch in tqdm(dataloader, desc="Batchs"):
            model.zero_grad()
            Uenc = model(batch, True)
            Nloss = Ncutsoft(Uenc, batch).to(device)
            print("Loss on this batch: {}".format(float(Nloss)))
            print("Accuracy on this batch : {}%".format(round(100*(1-float(Nloss)/K), 2)))
            Nloss.backward()
            optimizer.step()
            model.zero_grad()
            Udec = model(batch, False)[1]
            Jloss = Jrecons(Udec, batch).to(device)
            Jloss.backward()
            optimizer.step()
        torch.save({"model" : model.state_dict(),
                    "optimizer" : optimizer.state_dict()}, path_saved_models + '{}epoch'.format(i))
    return(model, optimizer)

# =============================================================================
# Dataset
# =============================================================================
checkpoint = None
K = 5
cpu = torch.device("cpu")
cuda = torch.device("cuda")
resolution = (64, 64)
sample = 3
path_data = "folder_with_images"
path_saved_models = "/modeles"
Namesimages = os.listdir(path_data)


# =============================================================================
# Main
# =============================================================================
def main():
    """
    Creates the data tensor and trainsss
    """
    data = torch.tensor([]).to(DEVICE)
    for name in Namesimages[:sample]:
        frames = io.imread(name)
        for i in range(len(frames)):
            frame = np.float32(frames[i])
            frame = frame/1024.0
            tens = torch.tensor(frame)
            tens = tens.unsqueeze_(0)
            tens = transforms.ToTensor()(transforms.Resize(resolution)(transforms.ToPILImage()(tens)))
            tens = tens.unsqueeze_(0).to(DEVICE)
            data = torch.cat((data, tens), 0)
    model, optimizer = learn(data, K,
                             resolution, EPOCH,
                             DEVICE, checkpoint=checkpoint)
    torch.save({"model" : model.state_dict(), "optimizer" : optimizer.state_dict()}, path_saved_models + 'trained_model')
    return print("fini")

if __name__ == '__main__':
    main()

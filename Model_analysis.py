"""
To try out a model on BSR for example
"""
from Segmenter import showimg
from Training import Ncutsoft
import scipy.io
import PIL as pl
from matplotlib import cm
import numpy as np
import torchvision.transforms as transforms
import torch
from skimage import io

resolution = (64, 64)
path_raw = "test_dataset_path"
path_truth = "groundtruth_path"

def ima(filename, n):
    """
    Resizes the image
    """
    ims = io.imread(filename)
    im = np.float32(ims[n])/1024.0
    im = transforms.ToPILImage()(torch.tensor(im).unsqueeze_(0))
    im = transforms.Resize(resolution)(im)
    return im



if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print('Initialisation de cuda')
    torch.cuda.init()
else:
    print('Mode CPU')
    DEVICE = torch.device('cpu')



def compare(model, filename):
    """
    Tests model on BSR dataset.
    Output : dictionnary containing raw image, groundtruth and model reconstructed image
    """
    filename = filename[:-4]
    dict = scipy.io.loadmat(path_truth+filename)
    mat = dict["groundTruth"]
    mat = mat[0, 4]['Segmentation'][0][0]
    im_truth = transforms.Resize(resolution)(pl.Image.fromarray(np.uint8(cm.gist_earth(mat)*255)))
    im_raw = transforms.Resize(resolution)(pl.Image.open(path_raw+filename+'.jpg'))
    tensor = transforms.ToTensor()(im_raw).unsqueeze_(0).to(DEVICE)
    im_model = showimg(model(tensor)[1])
    return {"raw": im_raw, "model" : im_model, "truth" : im_truth}

def seg(model, im):
    """
    Does the forward pass and gives all output
    Input : model and im in PIL format
    sortie : dictionarry containing raw, segmented and reconstructed image
    """
    im = transforms.Resize(resolution)(im)
    tensor = transforms.ToTensor()(im).unsqueeze_(0)
    tensor = tensor.to(DEVICE)
    Uenc, Udec = model(tensor)
    return {"raw" : tensor, "Enc" : Uenc, "Dec" : Udec}

def accuracyenc(model, dataeval):
    """
    Computes a value evaluating the ability of the model to group pixel close
    in colour and distance
    """
    cpu = torch.device("cpu")
    Uenc = model(dataeval, U=True).to(cpu)
    K = Uenc.size()[1]
    with torch.no_grad():
        accuracy = 1.0 - Ncutsoft(Uenc, dataeval.to(cpu), cpu)/K
    return "{}% of accuracy".format(round(float(100*accuracy), 2))

def discret_reconstruct(Uenc, K=3):
    """
    Transforms the segmented image tensor into pil image with a different colour
    for each segment
    """
    resolution = [3]
    resolution += list(Uenc.size()[2:])
    out = torch.zeros(resolution)
    labels = torch.argmax(Uenc[0], 0)
    for i in range(resolution[1]):
        for j in range(resolution[2]):
            if labels[i][j] <= 2:
                out[labels[i][j]][i][j] = 1
            if labels[i][j] == 3:
                for k in range(3):
                    out[k][i][j] = 1
    return out.unsqueeze_(0)

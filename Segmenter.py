"""
W network
"""
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchvision.transforms as transforms
from Segmenter_atoms import inconv, down, up, outconv

softmax = nn.Softmax(dim=1)
printresolution = (224, 224)

def showimg(tensor):
    """
    Converts tensor to pil image and shows it
    """
    tensor = tensor[0]
    size = tensor.size()
    if size[0] not in (1, 3) or len(size) != 3:
        return print("mauvaise taille pour la conversion en img")
    else:
        return transforms.Resize(printresolution)(transforms.ToPILImage()(tensor.to(torch.device("cpu"))))


class Segmenter(nn.Module):
    """
    The network
    """
    def __init__(self, resolution, K, device):
        """
        K (int) : number of different part we want the network to divide images into
        """
        super(Segmenter, self).__init__()
        self.resolution = resolution
        self.K = K
        self.device = device
# =============================================================================
# Uenc
# =============================================================================
        self.module1 = inconv(1, 64).to(self.device)
        self.down12 = down(64, 128).to(self.device)
        self.down23 = down(128, 256).to(self.device)
        self.down34 = down(256, 512).to(self.device)
        self.down45 = down(512, 1024).to(self.device)
        self.up56 = up(1024, 512).to(self.device)
        self.up67 = up(512, 256).to(self.device)
        self.up78 = up(256, 128).to(self.device)
        self.up89 = up(128, 64).to(self.device)
        self.outenc = outconv(64, K).to(self.device)

# =============================================================================
# Udec
# =============================================================================
        self.module10 = inconv(K, 64).to(self.device)
        self.down1011 = down(64, 128).to(self.device)
        self.down1112 = down(128, 256).to(self.device)
        self.down1213 = down(256, 512).to(self.device)
        self.down1314 = down(512, 1024).to(self.device)
        self.up1415 = up(1024, 512).to(self.device)
        self.up1516 = up(512, 256).to(self.device)
        self.up1617 = up(256, 128).to(self.device)
        self.up1718 = up(128, 64).to(self.device)
        self.outdec = outconv(64, 1).to(self.device)
# =============================================================================
# Forward
# =============================================================================
    def forward(self, x, U=False):
        #Uenc
        x1 = self.module1(x)
        x2 = self.down12(x1)
        x3 = self.down23(x2)
        x4 = self.down34(x3)
        x = self.down45(x4)
        x = self.up56(x, x4)
        x = self.up67(x, x3)
        x = self.up78(x, x2)
        x = self.up89(x, x1)
        x = self.outenc(x)
        Uenc = softmax(x)
        if U:
            return Uenc
        #Udec
        x10 = self.module10(Uenc)
        x11 = self.down1011(x10)
        x12 = self.down1112(x11)
        x13 = self.down1213(x12)
        x = self.down1314(x13)
        x = self.up1415(x, x13)
        x = self.up1516(x, x12)
        x = self.up1617(x, x11)
        x = self.up1718(x, x10)
        x = self.outdec(x)
        return (Uenc, x)
        
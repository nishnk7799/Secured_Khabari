import torch
from model import Hide, Reveal
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import math
#Crop the picture and convert it to tensor form
def transforms_img(img):
    img = Image.open(img) #read image
    img = img.resize((256, 256))
    tensor = transforms.ToTensor()(img) #Convert image to tensor，
    tensor = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)
    return tensor
# Import the model
def moxing(reveal):
    reveal_net = Reveal()
    reveal_net.eval()
    # reveal_net.cuda()

    reveal_net.load_state_dict(torch.load(reveal, map_location=torch.device('cpu')))
    return reveal_net
#Model indicator loss function calculation
def norm(img1, img2):
    criterion = nn.MSELoss()
    loss_r = criterion(img1,img2)
    print("loss_r:" + str(loss_r.item()))
    SSIM_h=1-pytorch_msssim.ssim(img1,img2)
    print("SSIM_h:"+str(SSIM_h.item()))
    MS_SSIM_h=1-pytorch_msssim.ms_ssim(img1,img2)
    print("MS_SSIM_h:"+str(MS_SSIM_h.item()))
def psnr1(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100

    return 10 * math.log10(255.0 ** 2 / mse)

def s_first(stag):
    # MSE Loss SSIM MS_SSIM
    reveal='./epoch_290_reveal.pkl'
    reveal_net=moxing(reveal)

    output=stag
    output=transforms_img(output)

    reveal_secret = reveal_net(output)

    save_image(reveal_secret.cpu().data[:4],fp='./result1/reveal_secret.png')
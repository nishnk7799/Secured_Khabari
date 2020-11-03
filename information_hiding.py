import torch
from model import Hide, Reveal
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from ssim3 import SSIM,MS_SSIM
import cv2
import numpy as np
import math

#Crop the picture and convert it to tensor form
def transforms_img(img):
    img = Image.open(img) # read image
    img = img.resize((256, 256))
    tensor = transforms.ToTensor()(img) # Convert the picture into tensor,
    tensor = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)
    return tensor
# Import the model
def moxing(hide):
    hide_net = Hide()
    hide_net.eval()

    hide_net.load_state_dict(torch.load(hide, map_location=torch.device('cpu')))
    return hide_net
#Model indicator loss function calculation
def norm(secret,cover):
    ssim=SSIM()
    loss=1-ssim(secret,cover)
    print(loss)

def psnr1(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)

def ih_first(sec,cov):
    # MSE Loss SSIM MS_SSIM
    hide='./epoch_290_hide.pkl'
    hide_net=moxing(hide)

    secret_img=sec
    cover_img=cov

    secret=transforms_img(secret_img)
    cover=transforms_img(cover_img)

    output = hide_net(secret, cover)
    norm(cover,cover)


    save_image(secret.cpu().data[:4],fp='./result1/secret.png')
    save_image(cover.cpu().data[:4],fp='./result1/cover.png')
    save_image(output.cpu().data[:4],fp='./result1/output.png')

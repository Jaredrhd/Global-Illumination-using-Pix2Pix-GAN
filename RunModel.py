import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import trace
from numpy.core.numeric import full
from numpy.lib.type_check import iscomplexobj
from torchvision import transforms
from torch.autograd import Variable
from torchvision.transforms.functional import InterpolationMode
import skimage
from Network import *
from DataHelper import *
from torch.utils.data import DataLoader
from ReadData import DataMap
from torchvision.utils import save_image
import time, datetime
from skimage.metrics import structural_similarity
import Helpers as hp
import scipy
cudaAvailable = True if torch.cuda.is_available() else False

#Network
generator = GeneratorUNet(12, 3)
discriminator = Discriminator(3 * 5)

if cudaAvailable:
    generator = generator.cuda()
    discriminator = discriminator.cuda()

loadModel = 71
generator.load_state_dict(torch.load("TRAINEDMODELS/generator_%d.pth" % loadModel), strict=False)
discriminator.load_state_dict(torch.load("TRAINEDMODELS/discriminator_%d.pth" % loadModel))


transform = [
    transforms.Resize((512, 1024), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

Tensor = torch.cuda.FloatTensor if cudaAvailable else torch.FloatTensor
dataLoaderTesting = DataLoader(DataMap("TestingData", transform), batch_size=1, shuffle=False, num_workers=4)

def calculate_fid(image1, image2):

    mu1, sigma1 = image1.mean(axis=0), np.cov(image1, rowvar=False)
    mu2, sigma2 = image2.mean(axis=0), np.cov(image2, rowvar=False)

    ssdiff = np.sum((mu1-mu2)**2)
    covMean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if iscomplexobj(covMean):
        covMean = covMean.real
    return ssdiff + trace(sigma1+sigma2-2*covMean)


if __name__ == '__main__':
    times = []

    L1LossDiffuse = []
    L1LossGenerated = []
    L1LossRI = []


    L2LossDiffuse = []
    L2LossGenerated = []
    L2LossRI = []

    SSIMLossDiffuse = []
    SSIMLossGenerated = []
    SSIMLossRI = []

    FIDLossDiffuse = []
    FIDLossGenerated = []
    FIDLossRI = []

    print("Cuda Available: " + str(cudaAvailable))
    for (i, image) in enumerate(dataLoaderTesting):

        depthImg =  Variable(image[0].type(Tensor))
        DIImg =  Variable(image[1].type(Tensor))
        DiffuseImg =  Variable(image[2].type(Tensor))
        NormalImg =  Variable(image[3].type(Tensor))
        RIImg =  Variable(image[4].type(Tensor))

        input = torch.cat((depthImg, DIImg, DiffuseImg, NormalImg), 1)

        prev_time = time.time()
        fake_B = generator(input)
        times.append(time.time() - prev_time)

        #Save image
        img_sample = torch.cat((DIImg.data, fake_B.data, RIImg.data), -2)
        img_sample = fake_B
        save_image(img_sample, "%s/%s.png" % ("TestOutput", i), nrow=1, normalize=True, value_range=(-1,1))
        
        img_sample = torch.cat((DIImg.data, RIImg.data), -2)
        save_image(img_sample, "%s/%s.png" % ("TestOutput", i), nrow=1, normalize=True)

        DIImg = DIImg.cpu().numpy()
        fake_B = fake_B.detach().cpu().numpy()
        RIImg = RIImg.cpu().numpy()

        DIImg = DIImg.transpose(0,2,3,1).squeeze()
        fake_B = fake_B.transpose(0,2,3,1).squeeze()
        RIImg = RIImg.transpose(0,2,3,1).squeeze()

        DIImg = hp.ContrastStretch(DIImg)
        fake_B = hp.ContrastStretch(fake_B)
        RIImg = hp.ContrastStretch(RIImg)



        #L1 losses
        L1LossDiffuse.append(np.linalg.norm(DIImg.ravel() - RIImg.ravel(), 1))
        L1LossGenerated.append(np.linalg.norm(fake_B.ravel() - RIImg.ravel(), 1))
        L1LossRI.append(np.linalg.norm(RIImg.ravel() - RIImg.ravel(), 1))


        #L2 losses
        L2LossDiffuse.append(np.linalg.norm(DIImg.ravel() - RIImg.ravel(),2)**2)
        L2LossGenerated.append(np.linalg.norm(fake_B.ravel() - RIImg.ravel(),2)**2)
        L2LossRI.append(np.linalg.norm(RIImg.ravel() - RIImg.ravel(),2)**2)


        #SSIM losses
        SSIMLossDiffuse.append(structural_similarity(RIImg, DIImg, multichannel=True))
        SSIMLossGenerated.append(structural_similarity(RIImg, fake_B, multichannel=True))
        SSIMLossRI.append(structural_similarity(RIImg, RIImg, multichannel=True))

        #FID losses

        FIDLossDiffuse.append(calculate_fid(RIImg.reshape(512*1024,3), DIImg.reshape(512*1024,3)))
        FIDLossGenerated.append(calculate_fid(RIImg.reshape(512*1024,3), fake_B.reshape(512*1024,3)))
        FIDLossRI.append(calculate_fid(RIImg.reshape(512*1024,3), RIImg.reshape(512*1024,3)))


        sys.stdout.write("\rCompleted {0} of {1}".format(i, 376))
        sys.stdout.flush()

    times = np.array(times)
    print("")
    print(np.average(times[1:]))
    print("")


    print(np.average(L1LossDiffuse))
    print(np.average(L1LossGenerated))
    print(np.average(L1LossRI))
    print("")

    print(np.average(L2LossDiffuse))
    print(np.average(L2LossGenerated))
    print(np.average(L2LossRI))
    print("")

    print(np.average(SSIMLossDiffuse))
    print(np.average(SSIMLossGenerated))
    print(np.average(SSIMLossRI))
    print("")

    print(np.average(FIDLossDiffuse))
    print(np.average(FIDLossGenerated))
    print(np.average(FIDLossRI))
    print("")


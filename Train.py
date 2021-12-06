import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from ReadData import DataMap
from Network import *
from DataHelper import *
import numpy as np
import time, datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image

#Is Cuda Available
cudaAvailable = True if torch.cuda.is_available() else False

#Loss functions
criterion_GAN = torch.nn.BCEWithLogitsLoss() #MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

#Network
generator = GeneratorUNet(12, 3)
discriminator = Discriminator(3 * 5)

if cudaAvailable:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

loadModel = -1

if loadModel >=0:
    generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % loadModel))
    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % loadModel))
else:
    generator.apply(weightsInitialize)
    discriminator.apply(weightsInitialize)

#Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

sd = SplitData("ImagesBlender", "TrainingData", "ValidationData", "TestingData")

#Uncomment the below to split the data
#sd.PerformSplit()

transform = [
    transforms.Resize((512, 1024), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataLoaderTraining = DataLoader(DataMap("TrainingData", transform), batch_size=1, shuffle=False, num_workers=3)
dataLoaderValidation = DataLoader(DataMap("ValidationData", transform), batch_size=1, shuffle=False, num_workers=1)
dataLoaderTesting = DataLoader(DataMap("TestingData", transform), batch_size=1, shuffle=False, num_workers=1)

# Tensor type
Tensor = torch.cuda.FloatTensor if cudaAvailable else torch.FloatTensor

lossWeight = 100
patchGanSize = (1, 512 // (2**4), 1024 // (2**4))

prev_time = time.time()

def sample_images(batches_done, image):
    torch.cuda.empty_cache()
    depthImg =  Variable(image[0].type(Tensor))
    DIImg =  Variable(image[1].type(Tensor))
    DiffuseImg =  Variable(image[2].type(Tensor))
    NormalImg =  Variable(image[3].type(Tensor))
    RIImg =  Variable(image[4].type(Tensor))
    #ShadowImg =  Variable(image[5].type(Tensor))
    
    #Input into generator
    input = torch.cat((depthImg, DIImg, DiffuseImg, NormalImg), 1)
    fake_B = generator(input)
    img_sample = torch.cat((DIImg.data, fake_B.data, RIImg.data), -2)
    save_image(img_sample, "%s/%s.png" % ("GenOutput", batches_done), nrow=1, normalize=True, value_range=(-1,1))


if __name__ == '__main__':
    print("Cuda Available: " + str(cudaAvailable))
    valIter = iter(dataLoaderValidation)
    totalEpochs = 2000
    for epoch in range(0, totalEpochs):
        for (i, image) in enumerate(dataLoaderTraining):
            
            depthImg =  Variable(image[0].type(Tensor))
            DIImg =  Variable(image[1].type(Tensor))
            DiffuseImg =  Variable(image[2].type(Tensor))
            NormalImg =  Variable(image[3].type(Tensor))
            RIImg =  Variable(image[4].type(Tensor))
            #ShadowImg =  Variable(image[5].type(Tensor))

            input = torch.cat((depthImg, DIImg, DiffuseImg, NormalImg), 1)

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((depthImg.size(0), *patchGanSize))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((depthImg.size(0), *patchGanSize))), requires_grad=False)


            #--------------------
            # Train Generators
            #--------------------

            optimizer_G.zero_grad()

            fake_B = generator(input)
            pred_fake = discriminator(fake_B.detach(), input)

            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_pixel = criterion_pixelwise(fake_B, RIImg)
            

            #Total loss
            loss_G = loss_GAN + lossWeight * loss_pixel

            loss_G.backward()
            optimizer_G.step()

            #--------------------
            #Train Discriminator
            #--------------------

            optimizer_D.zero_grad()

            #Real Loss
            pred_real = discriminator(RIImg, input)
            loss_real = criterion_GAN(pred_real, valid)

            #Fake Loss
            pred_fake = discriminator(fake_B.detach(), input)
            loss_fake = criterion_GAN(pred_fake, fake)

            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_D.step()


            #--------------------
            # Progress
            #--------------------

            batches_done = epoch * len(dataLoaderTraining) + i
            batches_left = totalEpochs * len(dataLoaderTraining) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            print("\r[Epoch {0}/{1}] [Batch {2}/{3}] [D loss: {4}] [G loss: {5}, pixel: {6}, adv: {7}] ETA: {8}".format(
                epoch,
                totalEpochs,
                i,
                len(dataLoaderTraining),
                loss_D.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_GAN.item(),
                time_left,
            ))

            if batches_done % 1000 == 0:
                image = next(valIter, None)

                #If we have looped over all validation, reset
                if image is None:
                    valIter = iter(dataLoaderValidation)
                    image = next(valIter, None)

                sample_images(batches_done, image)


            if batches_done % 1757 == 0:
                # Save model checkpoints
                torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % (epoch//20))
                torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % (epoch//20))

        

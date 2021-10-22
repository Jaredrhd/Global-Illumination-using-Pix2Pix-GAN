import os
import torchvision.transforms as transforms
from PIL import Image

class DataMap():
    #Give the directory and imageSet Start and End. 6 images makes 1 set, so 6*1000 = 6000 images
    def __init__(self, imageDirectory, transform, imageExtension = '.png'):
        self.transform = transforms.Compose(transform)
        self.depthImages = []
        self.directIlluminationImages = []
        self.diffuseImages = []
        self.normalImages = []
        self.rayTracedImages = []
        self.shadowImages = []

        files = sorted(os.listdir(imageDirectory))
        for fileName in files:
            if "DepthNorm" in fileName :
                self.depthImages.append(os.path.join(imageDirectory, fileName))
            elif "DI" in fileName:
                self.directIlluminationImages.append(os.path.join(imageDirectory, fileName))
            elif "Diffuse" in fileName:
                self.diffuseImages.append(os.path.join(imageDirectory, fileName))
            elif "Normal" in fileName:
                self.normalImages.append(os.path.join(imageDirectory, fileName))
            elif "RI" in fileName:
                self.rayTracedImages.append(os.path.join(imageDirectory, fileName))
            elif "Shadow" in fileName:
                self.shadowImages.append(os.path.join(imageDirectory, fileName))


    def __getitem__(self, index):
        depth = Image.open(self.depthImages[index])
        direct = Image.open(self.directIlluminationImages[index])
        diffuse = Image.open(self.diffuseImages[index])
        normal = Image.open(self.normalImages[index])
        rayTraced = Image.open(self.rayTracedImages[index])
        shadow = Image.open(self.shadowImages[index])

        depth = self.transform(depth)
        direct = self.transform(direct)
        diffuse = self.transform(diffuse) 
        normal = self.transform(normal)
        rayTraced = self.transform(rayTraced)
        shadow = self.transform(shadow)


        return depth, direct, diffuse, normal, rayTraced, shadow 

    def __len__(self):
        return len(self.directIlluminationImages)
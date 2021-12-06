import os
import numpy as np
from numpy.lib.function_base import diff
from torch.functional import norm
from shutil import copy2, rmtree

class SplitData():
    def __init__(self, imageDirectory, trainingDirectory, validationDirectory, testingDirectory, trainingPer = 0.70, valPer=0.15, testPer=0.15, imageExtension=".png"):
        
        self.trainingDirectory = trainingDirectory
        self.validationDirectory = validationDirectory
        self.testingDirectory = testingDirectory

        self.trainingPer = trainingPer
        self.valPer = self.trainingPer + valPer
        self.testPer = self.valPer + testPer

        self.depthImages = []
        self.directIlluminationImages = []
        self.diffuseImages = []
        self.normalImages = []
        self.rayTracedImages = []
        self.shadowImages = []

        if (os.path.isdir(imageDirectory)):
            filesCount = os.listdir(imageDirectory)
            filesCount = len(filesCount)

            for i in range(filesCount//7):
                indexString = str(i).zfill(4)
                self.depthImages.append(os.path.join(imageDirectory, indexString + "_DepthNorm" + imageExtension))
                self.directIlluminationImages.append(os.path.join(imageDirectory, indexString + "_DI" + imageExtension))
                self.diffuseImages.append(os.path.join(imageDirectory, indexString + "_Diffuse" + imageExtension))
                self.normalImages.append(os.path.join(imageDirectory, indexString + "_Normal" + imageExtension))
                self.rayTracedImages.append(os.path.join(imageDirectory, indexString + "_RI" + imageExtension))
                self.shadowImages.append(os.path.join(imageDirectory, indexString + "_Shadow" + imageExtension))
            
        if(not os.path.isdir("saved_models")):
            os.makedirs("saved_models")
        if(not os.path.isdir("GenOutput")):
            os.makedirs("GenOutput")


    def PerformSplit(self):
        #Merge Lists to split
        depth, direct, diffuse, normal, ray, shadow = np.array(self.depthImages), np.array(self.directIlluminationImages), np.array(self.diffuseImages), np.array(self.normalImages), np.array(self.rayTracedImages), np.array(self.shadowImages)
        shuffleMatrix = np.c_[depth, direct, diffuse, normal, ray, shadow]

        np.random.shuffle(shuffleMatrix)

        #Delete directories if they exist
        if(os.path.isdir(self.trainingDirectory)):
            rmtree(self.trainingDirectory)

        if(os.path.isdir(self.validationDirectory)):
            rmtree(self.validationDirectory)

        if(os.path.isdir(self.testingDirectory)):
            rmtree(self.testingDirectory)

        #Recreate directories
        os.makedirs(self.trainingDirectory)
        os.makedirs(self.validationDirectory)
        os.makedirs(self.testingDirectory)

        for i in range(len(shuffleMatrix)):
            for j in range(6):
                if i < len(shuffleMatrix) * self.trainingPer:
                    copy2(shuffleMatrix[i][j], self.trainingDirectory)
                elif len(shuffleMatrix) * self.trainingPer <= i <= len(shuffleMatrix) * self.valPer:
                    copy2(shuffleMatrix[i][j], self.validationDirectory)
                elif len(shuffleMatrix) * self.valPer <= i <= len(shuffleMatrix) * self.testPer:
                    copy2(shuffleMatrix[i][j], self.testingDirectory)
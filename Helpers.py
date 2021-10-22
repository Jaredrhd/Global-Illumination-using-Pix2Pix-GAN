import numpy as np
import skimage
import cv2
import math
from skimage import feature

#Contrast Stretch
def ContrastStretch(f, K=1):
    fMin = np.min(f, axis=(0,1))
    fMax = np.max(f, axis=(0,1))

    fs = K * ((f - fMin) / (fMax-fMin))
    return fs

#HISTOGRAMS
def GetBins(f, L = 256):
    return np.histogram(f, L)[0]

def IntensityShift(f, L = 256):
    bins = GetBins(f, L)
    prob = bins/f.size
    cdf = np.cumsum(prob)
    eq = cdf * (L-1)
    return eq

def HistogramEqual(f):
    fBins = IntensityShift(f).astype(np.uint8)
    return fBins[f]


#Read in images
def ReadImage(path, gray=False):
    if not gray:
        return skimage.img_as_float(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
    else:
        return skimage.img_as_float(cv2.imread(path, cv2.IMREAD_GRAYSCALE))

#Accuracy Function
def Accuracy(image, mask):
    return 1 - 1/(image.shape[0] * image.shape[1]) * np.sum(np.abs(image-mask))
    
#Filter Banks
def CreateCenteredMesh(shape=[3,3]):
    #Get values
    x_values = np.linspace(0, shape[0]-1, shape[0]) - (shape[0]//2)
    y_values = np.linspace(0, shape[1]-1, shape[1]) - (shape[1]//2)

    #Return Mesh
    return np.meshgrid(x_values, y_values)

def Preweitt(vertical=True):
    if vertical:
        return np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    else:
        return np.array([[1,0,-1],[1,0,-1],[1,0,-1]])

def Laplacian(diagonal=False):
    if diagonal:
        return np.array([[1,1,1],[1,-8,1],[1,1,1]])
    else:
        return np.array([[0,1,0],[1,-4,1],[0,1,0]])

def Sobel(vertical=True):
    if vertical:
        return np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    else:
        return np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

def GaussianKernel(shape=[3,3], sigma=6):
    mesh = CreateCenteredMesh(shape)
    return 1/(2*math.pi * sigma**2) *  np.exp(-(mesh[0]**2 + mesh[1]**2)/(2*sigma**2))

def LoG(shape=[3,3], sigma=6):
    mesh = CreateCenteredMesh(shape)
    return (-1/(math.pi*sigma**4)) * (1 - (mesh[0]**2 + mesh[1]**2)/(2*sigma**2)) * np.exp(-(mesh[0]**2 + mesh[1]**2)/(2*sigma**2))

def DoG(K, shape=[3,3], sigma=6):
    K = float(K)
    mesh = CreateCenteredMesh(shape)
    return 1/(2*math.pi*sigma**2) * np.exp(-(mesh[0]**2+mesh[1]**2)/(2*sigma**2))       -       1/(2*math.pi**K**2*sigma**2) * np.exp(-(mesh[0]**2+mesh[1]**2)/(2**K**2*sigma**2))


def GaussianEdgeBar(theta, shape=[3,3], sigmaXY=(6,6), edge=True):
    sigmaX, sigmaY = sigmaXY
    f = lambda x, sigma :  1/(np.sqrt(2*math.pi)* sigma) * np.exp(-x**2/(2*sigma**2))
    fPrime = lambda x, sigma :  f(x,sigma) * (-x/sigma**2)
    fDPrime = lambda x, sigma :  f(x,sigma) * ((x**2-sigma**2)/sigma**4)

    #Create Mesh
    mesh = CreateCenteredMesh(shape)

    #Apply rotation and formula
    xRot = np.cos(theta) * mesh[0] - np.sin(theta) * mesh[1]
    yRot = np.sin(theta) * mesh[0] + np.cos(theta) * mesh[1]

    if edge:
        return f(xRot, sigmaX) * fPrime(yRot, sigmaY)
    else:
        return f(xRot, sigmaX) * fDPrime(yRot, sigmaY)

def RFS():
    shape = [49,49]
    thetas = [i/6 * math.pi for i in range(6)]
    sigmaXY = [(3,1),(6,2),(12,4)]
    sigma= np.sqrt(10)
    filterBank = [GaussianKernel(shape, sigma), LoG(shape, sigma)]

    for i in range(2):
        for j in range(len(sigmaXY)):
            for k in range(len(thetas)):
                filterBank.append(np.flipud(GaussianEdgeBar(thetas[k], shape, sigmaXY[j], i==0)))

    return np.array(filterBank)

def MR8(image):
    allFilters = RFS()
    convolvedImages = []

    #Convolve all images
    for filter in allFilters:
        convolvedImages.append(cv2.filter2D(image, -1, filter))

    #Get RFS into a bank
    RFSImages = np.array(convolvedImages[2:]).reshape((6,6, 450,600,3))
    MR8 = np.max(RFSImages, axis=1)
    MR8 = np.concatenate((MR8, convolvedImages[:2]))

    #Remove shape of size 1s (24,450,600,1)=(24,450,600)
    MR8 = np.squeeze(np.concatenate(np.split(MR8, 3, axis=3), 0))

    return MR8

def LBP(image, radius = [4, 8, 16, 24, 32], rot="default"):
    #Features to return
    lbp = []

    #Split into RGB
    image = cv2.split(image)
    
    #Get LBP and return
    for j in radius:
        for i in image:
            lbp.append(feature.local_binary_pattern(i, 12, j, rot))

    lbp = np.array(lbp)
    return lbp.reshape(lbp.shape[0], lbp.shape[1] * lbp.shape[2])

#Calculates the integral image
def IntegralImage(image):
    #Define output and get the starting cols + rows
    integralImage = np.zeros(image.shape)
    integralImage[0,:] = np.cumsum(image[0,:], axis = 0)
    integralImage[:,0] = np.cumsum(image[:,0], axis = 0)
    
    #Calculate remaining points
    for i in range(1, integralImage.shape[0]):
        for j in range(1, integralImage.shape[1]):
            integralImage[i,j] = image[i,j] + integralImage[i-1,j] + integralImage[i,j-1] - integralImage[i-1,j-1]

    return integralImage

def HaarFilterConv(integralImage, filterSize):
    #Construct kernel
    kernel = -np.ones((filterSize, filterSize))
    kernel[:filterSize//2, :filterSize//2] = 1
    kernel[filterSize//2:, filterSize//2:] = 1

    #Get Pad size and apply edge pad
    padSize = filterSize//2 + 1
    paddedImage = np.pad(integralImage, padSize, 'edge')

    #Remove left and top padding, this is as if we integralled the padded image
    paddedImage[:padSize,:] = 0
    paddedImage[:,:padSize] = 0

    #Apply convolution
    kRight=filterSize//2 - 1
    kLeft=filterSize//2 + 1
    
    outputImage = np.zeros((integralImage.shape[0] + filterSize+2, integralImage.shape[1] + filterSize+2))
    for y in range(filterSize//2+1, integralImage.shape[0]+filterSize//2+1):
        for x in range(filterSize//2+1, integralImage.shape[1]+filterSize//2+1):

            #Get the BR, TL, TR, BL
            haarValue1 = (paddedImage[y+kRight, x+kRight] + paddedImage[y-1, x-1] - paddedImage[y-1, x+kRight] - paddedImage[y+kRight, x-1]) #BR
            haarValue2 = (paddedImage[y-1, x-1] + paddedImage[y - kLeft,  x-kLeft] - paddedImage[y-kLeft, x-1] - paddedImage[y-1, x-kLeft]) #TL
            haarValue3 = (paddedImage[y-1,  x+kRight] + paddedImage[y-kLeft, x-1] - paddedImage[y-kLeft, x+kRight] - paddedImage[y-1, x-1]) #TR
            haarValue4 = (paddedImage[y+kRight,  x-1] + paddedImage[y-1, x-kLeft] - paddedImage[y-1, x-1] - paddedImage[y+kRight, x-kLeft]) #TR

            outputImage[y,x] = (haarValue1 + haarValue2 - haarValue3 - haarValue4)#/(filterSize**2)

    return outputImage[padSize:-padSize,padSize:-padSize]

def HaarFilter(image, filterSize=[4,8,16]):
    haarFeatures = []
    integralChannels = []
    
    #Save integral images
    for j in cv2.split(image):
        integralChannels.append(IntegralImage(j))
    
    #Get haar filters
    for i in filterSize:
        for j in integralChannels:
            haarFeatures.append(HaarFilterConv(j, i))
    haarFeatures = np.array(haarFeatures)
    return haarFeatures.reshape(haarFeatures.shape[0], haarFeatures.shape[1]*haarFeatures.shape[2])

def ApplyFilters(image, filters = [Preweitt(True), Preweitt(False), Laplacian()]):
    features = []
    for i in filters:
        features.append(cv2.filter2D(image, -1, i))
    
    features = np.array(features)
    features = np.squeeze(np.concatenate(np.split(features, 3, axis=3), 0))
    return features.reshape(features.shape[0], features.shape[1]*features.shape[2])
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math

def im2col_sliding_strided(A, BSZ, stepsize=1):
    # Parameters
    m,n = A.shape
    s0, s1 = A.strides    
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1
 
    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::stepsize]


def testIm2Col():
    A = np.reshape(np.linspace(0,1,16),[4, 4]).T
    print(A)
    szKer = [2,2]
    print(im2col_sliding_strided(A,szKer))

def show2imgs(img1,img2):
    plt.subplot(121),plt.imshow(img1),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img2),plt.title('Averaging')
    plt.xticks([]), plt.yticks([])
    plt.show()


def genKern(Blur,Clean,szKer):

    
    [r,c] = Blur.shape
    Cut1 = math.ceil(szKer[0]/2.0)-1 
    Cut2 = math.floor(szKer[0]/2.0) 

    BlurCut = Blur[Cut1:r-Cut2, Cut1:c-Cut2]
    BlurCut = (np.ndarray.flatten(BlurCut))

    #BlurCut = BlurCut.reshape((BlurCut.shape)[0],1)
    # show2imgs(Blur,BlurCut)
    # print(Blur.shape,BlurCut.shape)

    cleanConv = im2col_sliding_strided(Clean,szKer).T

    # print(cleanConv.shape, BlurCut.shape)


    kern = np.linalg.lstsq(cleanConv,BlurCut)[0]
    kern = kern.reshape((szKer[0],szKer[1]))
    kern = cv.rotate(kern, cv.ROTATE_180)

    # print(kern)
    # # vXcorrKer = mGconv \ imBvalid(:)

    return kern


def plotImages(Clean,Blur,kernGen,recGen):
    plt.subplot(221),plt.imshow(Clean),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(Blur),plt.title('With Blur')
    plt.xticks([]), plt.yticks([])

    plt.subplot(223),plt.imshow(Clean),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(224),plt.imshow(recGen),plt.title('Blur with Estimate Kernal')
    plt.xticks([]), plt.yticks([])
    plt.show()

    plt.subplot(121),plt.imshow(kernGen),plt.title('Estimated Kernal')
    plt.xticks([]), plt.yticks([])

    # threshCut = np.mean(kernGen)
    # print(threshCut)

    threshCut = (np.max(kernGen)- np.min(kernGen))*.2 + np.min(kernGen)
    print(threshCut)


    ret1,thresh = cv.threshold(kernGen,threshCut,255,cv.THRESH_BINARY)
    plt.subplot(122),plt.imshow(thresh),plt.title('Estimated Kernal Threshold')
    plt.xticks([]), plt.yticks([])
    plt.show()

  


    






# img = cv.imread('BlurKernal/images/tommy.jpg')
# Clean = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# szKer = [11,11]
# kernel = np.ones((11,11),np.float32)/121

# Blur = cv.GaussianBlur(Clean,(11,11),0)
# # Blur = cv.filter2D(Clean,-1,kernel)

ImageIndex = range(918,932)

for i in ImageIndex:
    img1 = cv.imread('blurred_sharp/blurred_sharp/sharp/%d.png' % i)
    img2 = cv.imread('blurred_sharp/blurred_sharp/blurred/%d.png' % i)

    Clean = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    Blur = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    
    szKer = [33,33]
    # szKer = [15,15]
    kernGen = genKern(Blur,Clean,szKer)
    recGen = cv.filter2D(Clean,-1,kernGen)
    
    norm = cv.normalize(kernGen, None,alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    norm.astype(np.uint8)
    # plotImages(Clean,Blur,norm,recGen)

 
    # resize image
    resized = cv.resize(norm, (600, 600), interpolation = cv.INTER_AREA) 

    cv.imwrite('BlurKernal/Kernals/%d.png' % i, resized)


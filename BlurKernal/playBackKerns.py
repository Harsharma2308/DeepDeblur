import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math



img1 = cv.imread('BlurKernal/Kernals/%d.png' % 2)
img2 = cv.imread('blurred_sharp/blurred_sharp/blurred/%d.png' % 2)

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out1 = cv.VideoWriter('Kernal.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (img1.shape[0],img1.shape[1]))
out2 = cv.VideoWriter('Blur.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (img2.shape[0],img2.shape[1]))



ImageIndex = range(2,200)

for i in ImageIndex:
    img1 = cv.imread('BlurKernal/Kernals/%d.png' % i)
    img2 = cv.imread('blurred_sharp/blurred_sharp/blurred/%d.png' % i)
    

    out1.write(img1)
    out2.write(img2)

    cv.imshow('kernal',img1)
    cv.imshow('image',img2)

    if i == 2:
        print('wait')
        cv.waitKey(0)

    cv.waitKey(100)
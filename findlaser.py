import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize
import numpy as np
import math


def displayimg(image):
    plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
    plt.show()
    cv2.waitKey(0)


def applyclahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image


def getskeleton(image):
    red = image[:,:,2]
    green = image[:,:,1]
    blue = image[:,:,0]
    # print(red)
    # print(blue)

    redminblue = cv2.subtract(red,blue)
    # print(redminblue)
    redminblue = redminblue.dot(1.6)
    # print(redminblue)
    redmingreen = cv2.subtract(red,green)
    # print(type(redminblue))
    # print(type(redmingreen))
    nm = np.zeros(redminblue.shape)
    # print(nm.shape)
    for i in range(0, image.shape[0]): #to find minimum
        for j in range(0, image.shape[1]):
            if redminblue[i][j] >= redmingreen[i][j]:
                nm[i][j] = redmingreen[i][j]
            else:
                nm[i][j] = int(redminblue[i][j])

    # print(nm)

    ret,img = cv2.threshold(nm, 40, 255, cv2.THRESH_BINARY) #second parameter is the threshold

    img = (img - np.min(img))/np.ptp(img) #normalize image for skeletonize
    img = img.astype(np.uint8)
    skeleton = skeletonize(img) #takes an image(uint8) as input
    return skeleton.astype(np.uint8)*255


def applysaturation(img, amount):
    hsvImg = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #multiple by a factor to change the saturation
    hsvImg[...,1] = hsvImg[...,1]*amount

    #multiple by a factor of less than 1 to reduce the brightness
    #hsvImg[...,2] = hsvImg[...,2]*0.6

    img = cv2.cvtColor(hsvImg,cv2.COLOR_HSV2BGR)
    return img


image = cv2.imread('data/shadow/shadow (5).jpg')

# print(image.shape)
# cv2.imshow('window',image)
# cv2.waitKey(0)

#crop image
top = 450
bottom = 4450
left = 2450
right = 2780
img = image[top:bottom, left:right]

cv2.imwrite('results/before.jpg', img)

img = applysaturation(img, 2)
#img = applyclahe(img)

cv2.imwrite('results/after.jpg', img)

img = getskeleton(img)

cv2.imwrite('results/skeleton.jpg', img)
displayimg(img)

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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    binary = (binary - np.min(binary))/np.ptp(binary) #normalize image for skeletonize
    binary = binary.astype(np.uint8)
    skeleton = skeletonize(binary)
    return skeleton.astype(np.uint8)*255


def applysaturation(img, amount):
    hsvImg = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #multiple by a factor to change the saturation
    hsvImg[...,1] = hsvImg[...,1]*amount

    #multiple by a factor of less than 1 to reduce the brightness
    #hsvImg[...,2] = hsvImg[...,2]*0.6

    img = cv2.cvtColor(hsvImg,cv2.COLOR_HSV2BGR)
    return img


def getredinHSV(image):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    ## Gen lower mask (0-10) and upper mask (170-180) of RED
    mask1 = cv2.inRange(img_hsv, (0,20,70), (30,255,255))
    mask2 = cv2.inRange(img_hsv, (150,20,70), (180,255,255))

    ## Merge the mask and crop the red regions
    mask = cv2.bitwise_or(mask1, mask2)
    croped = cv2.bitwise_and(image, image, mask=mask)

    ## Display
    # cv2.imshow("mask", mask)
    # cv2.imshow("croped", croped)
    img_rgb = cv2.cvtColor(croped, cv2.COLOR_HSV2BGR)
    return croped
    # cv2.imwrite('results/hsv.jpg', croped)
    # cv2.waitKey()


def cleanimg(cimg):
    point = [1669, 154]
    while point[0] != 3030:
        if (cimg[point[0]+1][point[1]] == 255):
            # print("centre")
            point[0] += 1
            # print(point[0])
        elif (cimg[point[0]+1][point[1]-1] == 255):
            # print("left")
            point[0] += 1
            point[1] -= 1
            # print(point[0])
        elif (cimg[point[0]+1][point[1]+1] == 255):
            # print("right")
            point[0] += 1
            point[1] += 1
            # print(point[0])
        else:
            # print("point not found")
            point[0] += 1
            continue

        for i in range(0, 330):
            if i == point[1]:
                continue
            cimg[point[0]][i] = 0
    return cimg


image = cv2.imread('data/shadow/shadow (2).jpg')

# print(image.shape)
# cv2.imshow('window',image)
# cv2.waitKey(0)

#crop image
# for shadow 1-5
top = 450
bottom = 4450
left = 2450
right = 2780
# for shadow 17-19
# top = 1200
# bottom = 4607
# left = 2450
# right = 2730
img = image[top:bottom, left:right]

cv2.imwrite('results/before.jpg', img)

# img = applyclahe(img)
img = applysaturation(img, 2)
img = getredinHSV(img)
cv2.imwrite('results/after.jpg', img)
skeleton = getskeleton(img)
cv2.imwrite('results/skeleton.jpg', skeleton)
img = cleanimg(skeleton)
cv2.imwrite('results/clean.jpg', img)


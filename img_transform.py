
import torch
import cv2
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image

'''
This module is to define some functions related to the image transforms.

Img processing functions:
    - Loading and applying masks?: XXXXXXXXXXXX

    others:
    - rotate?
    - resize?

Note:
    - Size of input: (height,width,channels)  the return of cv2.imread()
    - the input of model should be (channels,height,width) (so I'd better use torchvision.transform)

Reference:
    1. 1st code https://www.kaggle.com/c/humpback-whale-identification/discussion/82366
    2. some code from image-augmentation https://www.kaggle.com/safavieh/image-augmentation-using-skimage
'''


def transforms_img():
    '''
    Just basic transform of the images.
    input: 
        PIL image data type
    return: 
        PIL to tensor
    '''
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        normalize
    ])
    return preprocess


def random_gaussian_noise(image, sigma=0.5):
    '''
    add random gaussian noise.

    But i'm not sure whether to use this
    '''
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray, a, b = cv2.split(lab)
    gray = gray.astype(np.float32)/255
    H, W = gray.shape

    noise = np.random.normal(0, sigma, (H, W))
    noisy = gray + noise

    noisy = (np.clip(noisy, 0, 1)*255).astype(np.uint8)
    lab = cv2.merge((noisy, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return image


def random_crop(im, p=0.5):
    '''
    croping the image
    '''
    if random.random() < p:
        margin = 1/4
        start = [int(random.uniform(0, im.shape[0] * margin)),
                 int(random.uniform(0, im.shape[1] * margin))]
        end = [int(random.uniform(im.shape[0] * (1-margin), im.shape[0])),
               int(random.uniform(im.shape[1] * (1-margin), im.shape[1]))]
        return im[start[0]:end[0], start[1]:end[1]]
    else:
        return im


def random_erase(image, p=0.5):
    '''
    It just erase a bit of space in the picture
    '''
    if random.random() < p:
        width, height, d = image.shape
        x = random.randint(0, width)
        y = random.randint(0, height)
        b_w = random.randint(0, 100)
        b_h = random.randint(0, 100)
        image[x:x+b_w, y:y+b_h] = 0
    return image


def random_affine(image):
    '''
    random implement the affine transform
    '''

    pass


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def random_angle_rotate(image, angles=[-30, 30]):
    '''
    I'm not sure whether the rotate will cause the performance, since after rotating, there are some black space in the image
    '''
    angle = random.randint(0, angles[1]-angles[0]) + angles[0]
    image = rotate(image, angle)
    return image


def random_horizintal_flip(image, p=0.5):
    '''
    Random decide whether the image should be flipped horizintal.

    You'd better not to use this, since it will confuse the learning system in this problem. As @earhian note in https://www.kaggle.com/c/humpback-whale-identification/discussion/82366.
    '''
    if random.random() < p:
        if len(image.shape) == 2:
            image = np.flip(image, 1)
        elif len(image.shape) == 3:
            image = np.flip(image, 1)
    return image


def add_mask(image, mask):
    '''
    Not all the images have mask, if they don't have, just add blank tensors
    '''
    pass


if __name__ == '__main__':
    IMG_PATH_TRAIN = "../Humpback-Whale-Identification-1st--master/input/train/"
    image_list = np.array(os.listdir(IMG_PATH_TRAIN))

    for i, img_name in enumerate(image_list):
        if i < 3:
            im = cv2.imread(IMG_PATH_TRAIN + img_name)
            # print(type(im))
            plt.imshow(im)
            plt.show()

            # # im_processed = random_crop(im_bbox)

            # 2. test adding gaussian noise
            im_processed = random_gaussian_noise(im, sigma=0.1)
            plt.imshow(im_processed)
            plt.show()

            # # 3. test random cropping
            # im_processed = random_crop(im)
            # plt.imshow(im_processed)
            # plt.show()

            # # 3. test random angle rotate
            # im_processed = random_angle_rotate(im)
            # plt.imshow(im_processed)
            # plt.show()

            # # 1. test basic transform
            # transform_train = transforms_img()
            # im_processed_2 = transform_train(Image.fromarray(im_processed))
            # print(im_processed_2.shape)
            # plt.imshow(np.transpose(np.array(im_processed_2), (1, 2, 0)))
            # plt.show()

            # # 3. test random erase
            # im_processed = random_erase(im)
            # plt.imshow(im_processed)
            # plt.show()

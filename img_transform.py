
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
    Just basic transform of the images
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


def random_gaussian_noise():
    '''
    random add gaussian noise
    '''
    pass


def random_crop(im):
    '''
    croping the image
    '''
    margin = 1/4
    start = [int(random.uniform(0, im.shape[0] * margin)),
             int(random.uniform(0, im.shape[1] * margin))]
    end = [int(random.uniform(im.shape[0] * (1-margin), im.shape[0])),
           int(random.uniform(im.shape[1] * (1-margin), im.shape[1]))]
    return im[start[0]:end[0], start[1]:end[1]]


def random_affine(image):
    '''

    '''
    pass


def random_rotate(image):
    '''
    '''
    pass


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

            # im_processed = random_crop(im_bbox)
            transform_train = transforms_img()
            im_processed = transform_train(Image.fromarray(im))
            print(im_processed.shape)

            plt.imshow(np.transpose(np.array(im_processed), (1, 2, 0)))
            plt.show()

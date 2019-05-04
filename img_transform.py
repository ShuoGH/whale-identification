
import torch
import cv2
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import pandas as pd
from skimage.transform import warp, AffineTransform, ProjectiveTransform
from torchvision import transforms

'''
This module is to define some functions related to the image transforms.

Img processing functions:
    - Loading and adding bounding box: `whaleData` module.
    - Loading and applying masks: XXXXXXXXXXXX

    others:
    - rotate?
    - resize?

Note:
    - Size of input: (height,width,channels)  the return of cv2.imread()

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
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])
    return preprocess


def load_bbox():
    '''
    Loading bounding box to locate whale tail
    '''
    bbox = pd.read_csv('./input/bboxs.csv')
    Images = bbox['Image'].tolist()
    x0s = bbox['x0'].tolist()
    y0s = bbox['y0'].tolist()
    x1s = bbox['x1'].tolist()
    y1s = bbox['y1'].tolist()
    bbox_dict = {}
    for Single_Image, x0, y0, x1, y1 in zip(Images, x0s, y0s, x1s, y1s):
        bbox_dict[Single_Image] = [x0, y0, x1, y1]
    return bbox_dict


def locate_bounding_box(image_name, image, bbox_dict):
    '''
    Input parameter:
        - image_name: as a key to find the bounding box edge
        - image: tensor of the image
        - bbox-dict: name-> (x0,y0,x1,y1) 
    '''
    x0, y0, x1, y1 = bbox_dict[image_name]
    im_bbox = image[int(y0):int(y1), int(x0):int(x1)]
    return im_bbox


def add_mask(image, mask):
    '''
    Not all the images have mask, if they don't have, just add blank tensors
    '''
    pass


def random_crop(im):
    '''
    croping the image
    '''
    margin = 1/5
    start = [int(random.uniform(0, im.shape[0] * margin)),
             int(random.uniform(0, im.shape[1] * margin))]
    end = [int(random.uniform(im.shape[0] * (1-margin), im.shape[0])),
           int(random.uniform(im.shape[1] * (1-margin), im.shape[1]))]
    return im[start[0]:end[0], start[1]:end[1]]


def random_affine(image):
    '''
    wrapper of Affine transformation with random scale, rotation, shear and translation parameters

    Note: 
        actually, when I have bounding box, I don't need to use the affine transform.
    '''
    tform = AffineTransform(scale=(random.uniform(0.75, 1.3), random.uniform(0.75, 1.3)),
                            rotation=random.uniform(-0.25, 0.25),
                            shear=random.uniform(-0.2, 0.2),
                            translation=(random.uniform(-im.shape[0]//10, im.shape[0]//10),
                                         random.uniform(-im.shape[1]//10, im.shape[1]//10)))
    return warp(image, tform.inverse, mode='reflect')


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


if __name__ == '__main__':
    IMG_PATH_TRAIN = "../Humpback-Whale-Identification-1st--master/input/train/"
    image_list = np.array(os.listdir(IMG_PATH_TRAIN))
    bbox_dict = load_bbox()

    for i, img_name in enumerate(image_list):
        if i < 3:
            im = cv2.imread(IMG_PATH_TRAIN + img_name)
            # print(type(im))
            plt.imshow(im)
            plt.show()

            # im_processed = transform_train(im)
            im_bbox = locate_bounding_box(img_name, im, bbox_dict)
            plt.imshow(im_bbox)
            plt.show()
            im_processed = random_crop(im_bbox)
            # print(im_processed.shape)
            plt.imshow(im_processed)
            plt.show()

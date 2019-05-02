# import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image


def basic_transform(img):
    '''
    Basic transform from https://www.kaggle.com/jhonatansilva31415/whales-a-simple-guide.
    '''
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    preprocess = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        # In this resnet, the input channel are 3, so keep the color channels
        # transforms.Resize(128),
        # transforms.CenterCrop(128),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    processed_img = preprocess(img)
    return processed_img

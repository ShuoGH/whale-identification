import matplotlib.pyplot as plt
import cv2
from PIL import Image
from whaleData import *
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
'''
Random choose and show an image in our data set.
'''


def show_image(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def read_image(path, img_name):
    # return cv2.imread(path + img_name, 0)
    return Image.open(path + img_name)


def test_snippet(dst_train):
    # Test and plot the images
    for i, content in enumerate(dst_train):
        if i < 3:
            img, _, = content
            show_image(img)


# def load_index_id(self):
#         # index_id_dict = {}
#     index_id_map_df = pd.read_csv("./input/label.csv")
#     # for i, row in index_id_map_df.iterrows():
#     #     index_id_dict[row['Id']] = row['Image']
#     # return index_id_dict
#     return index_id_map_df


if __name__ == '__main__':
    IMG_PATH_TRAIN = "../Humpback-Whale-Identification-1st--master/input/train/"
    IMG_PATH_TEST = "../Humpback-Whale-Identification-1st--master/input/test/"
    # img_name = "0037e7d3.jpg"
    data_train = pd.read_csv('./input/train.csv')
    names_train = data_train['Image'].values
    labels_train = data_train['Id'].values
    # index_id_dict = load_index_id()

    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transforms_1 = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       norm])
    dst_train = WhaleDatasetTrain(
        names_train, labels_train, transform_train=transforms_1)
    # show_image(read_image(IMG_PATH_TEST, img_name))
    test_snippet(dst_train)

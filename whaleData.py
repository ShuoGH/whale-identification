from torch.utils.data import Dataset
from PIL import Image
from torchvision.datasets.folder import default_loader
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
from img_transform import *
import img_transform


class WhaleDatasetTrain(Dataset):
    '''
    The data set of the Humpback Whale.
    To implement this data set:
        Put the data into ./input folder.

    names: name of image, include .jpg
    labels: `int` index of the whale id
    '''

    def __init__(self, names, labels=None, transform_train=None):
        super(WhaleDatasetTrain, self).__init__()
        self.names = names
        self.labels = labels
        self.img_bbox_dict = self.load_bbox()
        self.transform = transform_train  # implement transform

    def load_bbox(self):
        '''
        Load the bounding box to locate whale tails.
        '''
        # Image,x0,y0,x1,y1
        # print('loading bbox...')
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

    def __getitem__(self, img_index):
        '''
        According to index to get the image

        Return:
            PIL image/ ndarray image: (H,W,C)
        If you want input it into model, remember to convert it into (C,H,W)
        '''
        name = self.names[img_index]
        label = self.labels[img_index]

        im = cv2.imread(
            "../Humpback-Whale-Identification-1st--master/input/train/{}".format(name))
        try:
            x0, y0, x1, y1 = self.img_bbox_dict[name]
            im_bbox = im[int(y0):int(y1), int(x0):int(x1)
                         ]  # locate the whale tails
            im_processed = self.transform(im_bbox)
        except KeyError:
            im_processed = self.transform(im)

        return im_processed, label

    def __len__(self):
        return len(self.names)


class WhaleDatasetTest(Dataset):
    '''
    Test data of the whale tails.
    '''

    def __init__(self, names, transform_test=None):
        super(WhaleDatasetTest, self).__init__()
        self.names = names

        # self.transform_test = transform_test  # implement transform
        self.id_list = self.load_index_id()
        self.img_bbox_dict = self.load_bbox()
        self.transform = transform_test

    def load_index_id(self):
        '''
        Load csv file whick records the mapping between the id index and specific label.
        Use the list, we will map the output to string label name.
        '''
        index_id_map_df = pd.read_csv("./input/label.csv")
        # index_id_dict = {}
        # for i, row in index_id_map_df.iterrows():
        #     index_id_dict[row['Id']] = row['Image']
        # return index_id_dict
        return index_id_map_df['Image']

    def load_bbox(self):
        '''
        Loading bounding box to crop images to get better images.
        '''
        # Image,x0,y0,x1,y1
        # print('loading bbox...')
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

    def __getitem__(self, img_index):
        name = self.names[img_index]
        im = cv2.imread(
            "../Humpback-Whale-Identification-1st--master/input/test/{}".format(name))
        try:
            x0, y0, x1, y1 = self.img_bbox_dict[name]
            im_bbox = im[int(y0):int(y1), int(x0):int(x1)
                         ]  # locate the whale tails
            im_processed = self.transform(im_bbox)
        except KeyError:
            im_processed = self.transform(im)

        return im_processed

    def __len__(self):
        return len(self.names)


if __name__ == '__main__':
    # The following is just for testing

    def transform_train_img(img):
        '''
        input: cv2.imread image. 
        return: transformed PIL from torchvision.transform

        '''
        # do a series of transform on images
        img_processed = img_transform.random_gaussian_noise(img, sigma=0.1)
        img_processed = img_transform.random_angle_rotate(img_processed)
        img_processed = img_transform.random_crop(img_processed)
        img = Image.fromarray(img_processed)

        transform_basic = img_transform.transforms_img()
        return transform_basic(img)

    def transform_val_img(img):
        img = Image.fromarray(img)
        transform_basic = img_transform.transforms_img()
        return transform_basic(img)

    # IMG_PATH_TEST = "../Humpback-Whale-Identification-1st--master/input/test/"
    IMG_PATH_TRAIN = "../Humpback-Whale-Identification-1st--master/input/train/"
    # image_test_list = np.array(os.listdir(IMG_PATH_TEST))
    image_train_list = np.array(os.listdir(IMG_PATH_TRAIN))

    # dst_test = WhaleDatasetTest(image_test_list)
    test_casual_label = np.zeros(len(image_train_list))
    dst_train = WhaleDatasetTrain(
        image_train_list, test_casual_label, transform_train=transform_train_img)

    # for i, content in enumerate(dst_train):
    #     im, im_label = content
    #     if i < 4:
    #         # print(np.transpose(im, (1, 2, 0)).shape)
    #     plt.imshow(np.transpose(im, (1, 2, 0)))
    #     plt.show()
    #     print(im.shape)
    #     print(im_label)
    #     # print(im.shape)

    # # 2000+ id only have one image

    data_loader = torch.utils.data.DataLoader(
        dst_train, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    for i, j in data_loader:
        print(i.size(), j.size())

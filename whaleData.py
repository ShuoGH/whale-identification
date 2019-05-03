from torch.utils.data import Dataset
from PIL import Image
from torchvision.datasets.folder import default_loader
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os


class WhaleDatasetTrain(Dataset):
    '''
    The data set of the Humpback Whale.
    To implement this data set:
        Put the data into ./input folder.

    names: name of image, include .jpg
    labels: `int` index of the whale id
    '''

    def __init__(self, names, labels=None, transform_train=None, min_num_classes=0):
        super(WhaleDatasetTrain, self).__init__()
        self.names = names
        self.labels = labels
        self.img_bbox_dict = self.load_bbox()

        # self.transform_train = transform_train  # implement transform
        # self.transform_train = transform_train  # implement transform
        # self.names_id = {Image: Id for Image,
        #                  Id in zip(self.names, self.all_labels)}
        # # The following block isn's useful now
        # # save the images of the same whale and count the number of each whale
        # self.id_all_names = self.mapping_id_all_names()

        # # Use the min_num_classes to filter the whales with less pictures
        # self.filtered_labels = [k for k in self.id_all_names.keys()
        #                         if len(self.id_all_names[k]) >= min_num_classes]

    # def mapping_id_all_names(self):
    #     '''
    #     label: the id name of one whale
    #     name: image name
    #     '''
    #     id_all_names = {}
    #     for name, label in zip(self.names, self.all_labels):
    #         if label not in id_all_names.keys():
    #             id_all_names[label] = [name]
    #         else:
    #             id_all_names[label].append(name)
    #     return id_all_names

    def load_bbox(self):
        '''
        Load the bounding box to locate whale tails.
        '''
        # Image,x0,y0,x1,y1
        print('loading bbox...')
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
        '''
        name = self.names[img_index]
        label = self.labels[img_index]

        im = cv2.imread(
            "../Humpback-Whale-Identification-1st--master/input/train/{}".format(name))
        try:
            x0, y0, x1, y1 = self.img_bbox_dict[name]
            im_bbox = im[int(y0):int(y1), int(x0):int(x1)
                         ]  # locate the whale tails
            im_processed = cv2.resize(im_bbox, (224, 224))
        except KeyError:
            im_processed = cv2.resize(im, (224, 224))
        return im_processed, label
        # transformed_im = self.transform_train(im_bbox)

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
            im_processed = cv2.resize(im_bbox, (224, 224))
        except KeyError:
            im_processed = cv2.resize(im, (224, 224))
        return im_processed

    def __len__(self):
        return len(self.names)


if __name__ == '__main__':
    # IMG_PATH_TEST = "../Humpback-Whale-Identification-1st--master/input/test/"
    IMG_PATH_TRAIN = "../Humpback-Whale-Identification-1st--master/input/train/"
    # image_test_list = np.array(os.listdir(IMG_PATH_TEST))
    image_train_list = np.array(os.listdir(IMG_PATH_TRAIN))

    # dst_test = WhaleDatasetTest(image_test_list)
    test_casual_label = np.zeros(len(image_train_list))
    dst_train = WhaleDatasetTrain(image_train_list, test_casual_label)
    # for i, im in enumerate(dst_test):
    for i, content in enumerate(dst_train):
        im, _ = content
        if i < 4:
            plt.imshow(im)
            plt.show()
            print(im.shape)
            # print(im.shape)

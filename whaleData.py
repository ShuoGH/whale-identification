from torch.utils.data import Dataset
# import cv2
from PIL import Image
from torchvision.datasets.folder import default_loader
import numpy as np
import pandas as pd
import cv2


class WhaleDatasetTrain(Dataset):
    '''
    The data set of the Humpback Whale.
    To implement this data set:
        Put the data into ./input folder.

    names: the name of image, include .jpg
    labels: The label we transfer into is an int type data.
    '''

    def __init__(self, names, labels=None, transform_train=None, min_num_classes=0):
        super(WhaleDatasetTrain, self).__init__()
        self.names = names
        self.all_labels = labels

        self.transform_train = transform_train  # implement transform

        self.names_id = {Image: Id for Image,
                         Id in zip(self.names, self.all_labels)}
        # save the images of the same whale and count the number of each whale
        self.id_all_names = self.mapping_id_all_names()

        self.img_bbox_dict = self.load_bbox()

        # Use the min_num_classes to filter the whales with less pictures
        self.filtered_labels = [k for k in self.id_all_names.keys()
                                if len(self.id_all_names[k]) >= min_num_classes]

    def mapping_id_all_names(self):
        '''
        label: the id name of one whale
        name: image name
        '''
        id_all_names = {}
        for name, label in zip(self.names, self.all_labels):
            if label not in id_all_names.keys():
                id_all_names[label] = [name]
            else:
                id_all_names[label].append(name)
        return id_all_names

    def load_bbox(self):
        '''
        Loading bounding box to crop images to get better images.
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
        and return the transformed image tensor.
        '''
        name = self.names[img_index]
        label = self.names_id[name]
        # label_index = self.id_index[label]
        # im = Image.open(
        #     "../Humpback-Whale-Identification-1st--master/input/train/"+name)

        x0, y0, x1, y1 = self.img_bbox_dict[name]
        # im = default_loader(
        #     "../Humpback-Whale-Identification-1st--master/input/train/{}".format(name))
        im = cv2.imread(
            "../Humpback-Whale-Identification-1st--master/input/train/{}".format(name))

        im_bbox = im[int(y0):int(y1), int(x0):int(x1)]
        # transformed_im = self.transform_train(im_bbox)
        return im_bbox, label

    def __len__(self):
        return len(self.names)


class WhaleDatasetTest(Dataset):
    '''
    Test data of the whale tails.
    '''

    def __init__(self, names, transform_test=None):
        super(WhaleDatasetTest, self).__init__()
        self.names = names

        self.transform_test = transform_test  # implement transform
        self.id_list = self.load_index_id()

    def load_index_id(self):
        # index_id_dict = {}
        index_id_map_df = pd.read_csv("./input/label.csv")
        # for i, row in index_id_map_df.iterrows():
        #     index_id_dict[row['Id']] = row['Image']
        # return index_id_dict
        return index_id_map_df['Image']

    def __getitem__(self, img_index):
        '''
        According to index to get the image
        and return the transformed image tensor.
        '''
        name = self.names[img_index]
        im = self.transform_test(default_loader(
            "../Humpback-Whale-Identification-1st--master/input/test/{}".format(name)))

        return im

    def __len__(self):
        return len(self.names)

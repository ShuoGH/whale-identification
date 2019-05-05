import torch
from whaleData import *
from model import *
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
import copy
from tqdm import tqdm
import time
import img_transform
import random
'''
When you update the train.py to test more mdoel:
You need to update:
    - model.save(xxxxxx)

Try to set: 
    - another optimizer: Adam
'''


def load_label_index_dict():
    '''
    Return the dict of the mappinp:
        image label -> index id
    '''
    id_index_dict = {}
    index_id_map_df = pd.read_csv("./input/label.csv")
    for i, j in index_id_map_df.iterrows():
        id_index_dict[j['Image']] = j['Id']
    return id_index_dict


def load_oversampled_data():
    '''
    Load the oversampled train data from .csv file.
    return:
        - oversampled train data frame
        - 2931 val data: pick image from 2931 kinds of whales that own at least 2 images
    '''
    data_train = pd.read_csv('./input/oversampled_train_data.csv')
    data_val = pd.read_csv('./input/val_data.csv')
    return data_train, data_val


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


if __name__ == '__main__':
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    model_name_train = "resnet50"
    # model_name_train = "resnet101"

    # ---- load the oversampled data and selected val data
    data_train, data_val = load_oversampled_data()
    train_data_name = data_train['Image']
    val_data_name = data_val['Image']
    '''
    *****************
    reference:
        - @radek https://github.com/radekosmulski/whale
    '''

    # ---- convert the string label to int id ----
    id_index_dict = load_label_index_dict()  # dict: map label -> index id
    indexId_data_train = [id_index_dict[label]
                          for label in data_train['Id']]
    indexId_data_val = [id_index_dict[label]
                        for label in data_val['Id']]
    # ---- initialize train and val data----
    whale_train_data = WhaleDatasetTrain(
        train_data_name, indexId_data_train, transform_train=transform_train_img)
    whale_val_data = WhaleDatasetTrain(
        val_data_name, indexId_data_val, transform_train=transform_val_img)

    datasets_dict = {'train': whale_train_data, 'val': whale_val_data}

    dataloaders_dict = {phase: torch.utils.data.DataLoader(datasets_dict[phase], batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
                        for phase in ['train', 'val']}

    # test the pretrained model
    # Now there will no new_whale in our training phase
    # Only in test phase, I will give a threshold to tell which value is better to get higher LB score.
    num_classes = 5004

    '''
    end of the new demand of `no_new_whale` branch.
    '''

    model = model_whale(num_classes=num_classes,
                        inchannels=3, model_name=model_name_train).to(device)
    # model.freeze()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                       model.parameters()), lr=0.001, momentum=0.9)
    # optimiser = optim.Adam(
    #     filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    criterion = nn.CrossEntropyLoss()
    NUM_EPOCHS = 30

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    since = time.time()
    for epoch in range(NUM_EPOCHS):
        print('Epoch {}/{}'.format(epoch, NUM_EPOCHS - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            for X_batch, y_batch in tqdm(dataloaders_dict[phase]):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()

                # print(X_batch.size())
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * X_batch.size(0)
                running_corrects += torch.sum(preds == y_batch.data)

            epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = running_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

        print('\n')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best acc: {:.4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)

    torch.save(model.state_dict(),
               './trained_model/{}_5thMay_oversample_no_new_whale.model'.format(model_name_train))

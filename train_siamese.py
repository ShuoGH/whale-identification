import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Subset
from PIL import Image
import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
import time
import img_transform
import random
from sklearn.model_selection import ShuffleSplit
import losses_siamese as losses
from whale_data_siamese import SiameseTrainData
from networks_siamese import EmbeddingNet, SiameseNet
'''
This train_siamese.py script is for a better encoding method to get better image embeddings.

When you update the train_siamese.py to test more mdoel:
You need to update:
    - model.save(xxxxxx)

Try to set:
    - another optimizer: Adam
'''


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


if __name__ == '__main__':
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    BS = 32
    margin = 1
    # ---- initialize the train/val data set and load it----
    data_siamese = SiameseTrainData(transform_img=transform_train_img)

    # ---- split the siamese data into train and val data set----
    splitter = ShuffleSplit(n_splits=1, test_size=0.1)
    (train_idxs, val_idxs) = next(splitter.split(data_siamese))

    siamese_train_data = Subset(data_siamese, train_idxs)
    siamese_val_data = Subset(data_siamese, val_idxs)
    # ---- load train and val data set----
    dst_train_loader = torch.utils.data.DataLoader(
        siamese_train_data, batch_size=BS, shuffle=True, num_workers=0, pin_memory=True)
    dst_val_loader = torch.utils.data.DataLoader(
        siamese_val_data, batch_size=BS, shuffle=True, num_workers=0, pin_memory=True)
    dst_dict = {"train": dst_train_loader, "val": dst_val_loader}
    # print(len(siamese_train_data))

    # ---- initialize the Siamese Net, losses function and optimizer----
    model = SiameseNet(EmbeddingNet()).to(device)
    # print(model)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = losses.ContrastiveLoss(margin)

    NUM_EPOCHS = 15

    # --- start training and validating----
    val_acc_history = []

    # best_model_wts = copy.deepcopy(model.state_dict())

    since = time.time()
    for epoch in range(NUM_EPOCHS):
        print('Epoch {}/{}'.format(epoch, NUM_EPOCHS - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            # running_loss = 0.0
            # running_corrects = 0
            losses = []
            total_losses = 0
            for X_batch, y_batch in tqdm(dst_dict[phase]):
                im_a_batch = X_batch[0].to(device)
                im_b_batch = X_batch[1].to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                # print(X_batch.size())
                outputs_a, outputs_b = model(im_a_batch, im_b_batch)

                loss_outputs = criterion(outputs_a, outputs_b, y_batch)
                loss = loss_outputs[0] if type(loss_outputs) in (
                    tuple, list) else loss_outputs
                losses.append(loss)
                total_losses += loss
                loss.backward()
                optimizer.step()

            epoch_loss = total_losses / len(dst_dict[phase].dataset)
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())
            # if phase == 'val':
            #     val_acc_history.append(epoch_acc)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
        print('\n')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # print('Best acc: {:.4f}'.format(best_acc))
    # model.load_state_dict(best_model_wts)

    torch.save(model.state_dict(),
               './trained_model/Siamese_7thTrain.model')

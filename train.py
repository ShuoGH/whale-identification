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
'''
When you update the train.py to test more mdoel:
You need to update:
    - model.save(xxxxxx)
'''

if __name__ == '__main__':
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    # min_num_class = 0
    model_name_train = "resnet50"
    # model_name_train = "resnet101"

    data_train = pd.read_csv('./input/train.csv')
    names_train = data_train['Image'].values
    labels_train = data_train['Id'].values

    # ---- convert the string label to int id ----
    unique_labels_value = np.unique(labels_train)
    unique_label_index_dict = {}
    unique_index_label_dict = {}
    for i in range(len(unique_labels_value)):
        unique_label_index_dict[unique_labels_value[i]] = i
        unique_index_label_dict[i] = unique_labels_value[i]
    labelId_train = np.array([unique_label_index_dict[label]
                              for label in labels_train])

    # ---- Split the train and val data sets ----
    splitter = ShuffleSplit(n_splits=1, test_size=0.1)
    # splitter is a generator and the nun of iteration is only 1, so use next to get the result of splitting
    (train_idxs, val_idxs) = next(splitter.split(names_train, labelId_train))
    idxs = {'train': train_idxs, 'val': val_idxs}

    images_dict = {phase: names_train[idxs[phase]]
                   for phase in ['train', 'val']}
    labelsId_dict = {phase: labelId_train[idxs[phase]]
                     for phase in ['train', 'val']}
    # #  I will not use transform in this time 3th May
    # norm = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # transforms_dict = {
    #     'train': transforms.Compose([transforms.RandomResizedCrop(224),
    #                                  transforms.RandomHorizontalFlip(),
    #                                  transforms.ToTensor(),
    #                                  norm]),
    #     'val': transforms.Compose([transforms.Resize(224),
    #                                transforms.CenterCrop(224),
    #                                transforms.ToTensor(),
    #                                norm])
    # }
    datasets_dict = {phase: WhaleDatasetTrain(
        images_dict[phase], labelsId_dict[phase]) for phase in ['train', 'val']}
    dataloaders_dict = {phase: torch.utils.data.DataLoader(datasets_dict[phase], batch_size=32, shuffle=True, num_workers=1, pin_memory=True)
                        for phase in ['train', 'val']}

    # test the pretrained model
    num_classes = 5005
    model = model_whale(num_classes=num_classes,
                        inchannels=3, model_name=model_name_train).to(device)
    # model.freeze()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                       model.parameters()), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    NUM_EPOCHS = 30

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    since = time.time
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
               './trained_model/{}_3thMay.model'.format(model_name_train))

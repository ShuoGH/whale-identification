import torch
import os
from model import *
import torchvision.transforms as transforms
from whaleData import WhaleDatasetTest
import numpy as np
from tqdm import tqdm
import pandas as pd

'''
When you update the test.py to test more mdoel:
You need to update:
    - model.load(xxxxxx)
    - csv.to(xxxxxx)
'''
if __name__ == '__main__':
    device = "cuda:3" if torch.cuda.is_available() else "cpu"

    num_classes = 5005
    model_name = "resnet50"
    # model_name = "resnet101"

    model = model_whale(num_classes=num_classes,
                        inchannels=3, model_name=model_name).to(device)

    model.load_state_dict(torch.load(
        './trained_model/{}_4thMay_no_freeze.model'.format(model_name)))

    model.eval()

    # define a whale test data set, since the url path is written within the data set class
    TEST_PATH = "../Humpback-Whale-Identification-1st--master/input/test/"
    images_test = np.array(os.listdir(TEST_PATH))
    # print(len(images_test))
    dataset_test = WhaleDatasetTest(
        images_test)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=32, shuffle=False, num_workers=1, pin_memory=True)

    index_id = np.array(dataset_test.id_list)

    test_classnames = []
    for test_batch in tqdm(dataloader_test):
        test_batch = test_batch.to(device, dtype = torch.float)
        outputs = model(test_batch)
        predinds = torch.argsort(outputs, dim=1, descending=True)[:, :5]

        whale_labels = index_id[predinds.to('cpu').detach().numpy()].tolist()

        test_classnames.extend([" ".join(s) for s in whale_labels])

    testdf = pd.DataFrame({'Image': images_test, 'Id': test_classnames})
    testdf.to_csv(
        './results/submission_{}_4thMay_no_freeze.csv'.format(model_name), index=False)

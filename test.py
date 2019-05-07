import torch
import os
from model import *
import torchvision.transforms as transforms
from whaleData import WhaleDatasetTest
import numpy as np
from tqdm import tqdm
import pandas as pd
import img_transform

'''
When you update the test.py to test more mdoel:
You need to update:
    - model.load(xxxxxx)
    - csv.to(xxxxxx)


*******
You need to set proper threshold to classifi new_whale.
    - when you are doing testing
You also need to set the transform operation since you have edit it.
    - transform on the test whaledataset

In the training phase, the CrossEntropy have calculated the softmax value, I don't need to add the softmax layer in my model.
In the testing phase, if I want to get the probabilities of each class, then I need to add a `nn.Softmax` by my self.
'''


def transform_image(img):
    img = Image.fromarray(img)
    transform_basic = img_transform.transforms_img()
    return transform_basic(img)


if __name__ == '__main__':
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    # the train data we used have 5004 whale ids
    num_classes = 5004
    model_name = "resnet50"
    # model_name = "resnet101"

    model = model_whale(num_classes=num_classes,
                        inchannels=3, model_name=model_name).to(device)

    model.load_state_dict(torch.load(
        './trained_model/{}_5thMay_oversample_no_new_whale.model'.format(model_name)))

    model.eval()

    # define a whale test data set, since the url path is written within the data set class
    TEST_PATH = "../Humpback-Whale-Identification-1st--master/input/test/"

    images_test = np.array(os.listdir(TEST_PATH))
    # print(len(images_test))
    dataset_test = WhaleDatasetTest(
        images_test, transform_test=transform_image)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=32, shuffle=False, num_workers=1, pin_memory=True)

    index_id = np.array(dataset_test.id_list)

    test_classnames = []

    softmax_output = torch.nn.Softmax()
    for test_batch in tqdm(dataloader_test):
        test_batch = test_batch.to(device, dtype=torch.float)
        outputs = model(test_batch)
        '''
        Try the threshold with 0.9
        '''
        # Calculate the softmax output which represent the probability
        # Ouputs: 32*5004
        probability_outputs = softmax_output(outputs)
        detect_new_whale_index = [1 if torch.max(
            single_output) > 0.90 else 0 for single_output in probability_outputs]

        # predinds = torch.argsort(outputs, dim=1, descending=True)[:, :5]
        predinds = torch.argsort(
            probability_output, dim=1, descending=True)[:, :5].to('cpu').detach().numpy()

        # Assume the threshold of probability is 0.9
        predinds_result = np.array([[0] + j[:4] if i == 1 else j[:5]
                                    for i, j in zip(detect_new_whale_index, predinds)])

        whale_labels = index_id[predinds_result].tolist()

        test_classnames.extend([" ".join(s) for s in whale_labels])

    testdf = pd.DataFrame({'Image': images_test, 'Id': test_classnames})
    testdf.to_csv(
        './results/submission_{}_5thMay_oversample_no_new_whale.csv'.format(model_name), index=False)

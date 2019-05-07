# from PIL import Image
import img_transform
from networks_siamese import EmbeddingNet, SiameseNet
from whaleData import WhaleDatasetTest, WhaleDatasetTrain
from scipy.spatial import distance
import pickle
import pandas as pd
from train import load_label_index_dict
import torch
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

'''
Test strategy:
    Use the trained model based on the train data (it's a relative good way to encode the images)

    For each image in the test data set:
        - For each image in data set not new_whale, encode the images and calculate contrastive loss as the score.
        - For each whale from the training set, compute the score as the maximum score for this whale.
        - Add `new_whale` with a fixed new whale score of 'threshold'.
        - Sort the whale in the decreasing order.
    The first five whales are our outputs.
'''


def calculate_prediction(x1, y1, label_list=None):
    '''
    Use the Euclidean distance to get the score.
        x1: (32,256)
        y1: (25000+, 256)
        label_list: the list of labels (string)
    intermediate:
        cdist return (32,25000+) distance matrix

    return:
        - top_5_score_list: (32,5)
        - top_5_score_whale_labels_list: (32,5)

    Reference:
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    '''
    score_tensor = distance.cdist(x_i, x2)
    top_5_score_list = []
    top_5_score_whale_labels_list = []

    for score_single_img in score_tensor:
        score_df_single_img = pd.DataFrame(
            {"score": score_single_img, "id": label_list})
        score_group = score_df_single_img.groupby(
            ['id']).max().sort_value(by=['score'])['score']
        top_5_score_whale_labels = np.array(
            [whale_label for whale_label in score_group.index[:5]])

        top_5_scores = [score_im for score_im in score_group[:5]]

        top_5_score_list.append(top_5_scores)
        top_5_score_whale_labels_list.append(top_5_score_whale_labels)
    return top_5_score_list, top_5_score_whale_labels_list


def transform_image(img):
    img = Image.fromarray(img)
    transform_basic = img_transform.transforms_img()
    return transform_basic(img)


def get_train_data():
    '''
    return DataFrane
    '''
    return pd.read_csv("./input/train_no_new_whale.csv")


def get_all_embedding_training(model, whale_data_loader, preCalculated=True):
    '''
    load the pretrained embedding of all the training data set.

    Input parameter:
        model: the siamese model which is for encoding
        whale_data_loader: train data without new_whale
    return:  
        all_embedding_training (tensor: [len of train images * 256])
    '''
    if preCalculated:
        all_embedding_training = pickle.load(
            "./input/all_embedding_training_tensor.pl")
        return all_embedding_training
    else:
        all_embedding_training = torch.Tensor()
        print("Getting all the embedding of train images...")
        for im, label in tqdm(whale_data_loader):
            embedding_single_im = model.get_embedding(im)
            # print(embedding_single_im.size())
            all_embedding_training = torch.cat(
                (all_embedding_training, embedding_single_im), 0)
            # print(all_embedding_training.size())

        pickle.dump(all_embedding_training,
                    "./input/all_embedding_training_tensor.pl")
        return all_embedding_training


if __name__ == '__main__':
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    model = SiameseNet(EmbeddingNet()).to(device)
    model.load_state_dict(torch.load(
        './trained_model/Siamese_7thTrain.model'))

    model.eval()

    # define a whale test data set, since the url path is written within the data set class
    TEST_PATH = "../Humpback-Whale-Identification-1st--master/input/test/"
    images_test = np.array(os.listdir(TEST_PATH))
    # print(len(images_test))

    # ---- load all the embedding of training data set----
    dst_train = get_train_data()  # get the data frame of the train data

    id_index_dict = load_label_index_dict()  # dict: map whale_id -> index id
    indexId_data_train = [id_index_dict[label]
                          for label in dst_train['Id']]

    whale_train_data = WhaleDatasetTrain(
        dst_train['Image'], indexId_data_train, transform_train=transform_image)
    training_set_embedding = get_all_embedding_training(model, torch.utils.data.DataLoader(
        whale_train_data, batch_size=32, shuffle=False, num_workers=1, pin_memory=True), preCalculated=False)

    # ---- Load the test data set, we can use the previous whale data set----
    dataset_test = WhaleDatasetTest(
        images_test, transform_img)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=32, shuffle=False, num_workers=1, pin_memory=True)

    index_id = np.array(dataset_test.id_list)  # index -> whale_id str
    test_classnames = []  # store the results

    for test_batch in tqdm(dataloader_test):
        '''
        I think there will be a lot of bugs... 
        Since the in the second line return batch embedding.
        '''
        test_batch = test_batch.to(device, dtype=torch.float)
        embeddings_test = model.get_embedding(test_batch)
        print(embeddings_test.shape)  # (32,256)

        top_5_score_tensor, top_5_whale_labels = calculate_prediction(
            test_batch, training_set_embedding, label_list=dst_train['Id'])

        # ---- calculate the normalized top_5 matric so that can compare with the threshold----
        top_5_score_sum_1 = torch.sum(
            torch.tensor(top_5_score_tensor), 1)  # tensor

        top_5_normalized = top_5_score_tensor / \
            top_5_score_sum_1.reshape(
                top_5_score_tensor.size()[0], -1)  # tensor

        for index in range(top_5_normalized.size()[0]):
            if top_5_normalized[index, 0] < 0.1:
                top_5_whale_labels[index] = [
                    'new_whale'] + top_5_whale_labels[:4]

        test_classnames.extend([" ".join(s) for s in top_5_whale_labels])

    testdf = pd.DataFrame({'Image': images_test, 'Id': test_classnames})
    testdf.to_csv(
        './results/submission_{}_Siamese_7thTrain.csv'.format(model_name), index=False)

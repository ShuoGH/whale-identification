# from PIL import Image
import img_transform
from networks_siamese import EmbeddingNet, SiameseNet
from whaleData import WhaleDatasetTest, WhaleDatasetTrain
from scipy.spatial import distance
import pickle
import pandas as pd
from train import load_label_index_dict
import torch

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


def calculate_score(x1, y1):
    '''
    Use the Euclidean distance to get the score.
    '''
    score = distance.euclidean(x1, x2)
    return score


def get_train_data():
    '''
    return DataFrane
    '''
    return pd.read_csv("./input/train_no_new_whale.csv")


def get_all_embedding_training(model, whale_data, preCalculated=True):
    '''
    load the pretrained embedding of all the training data set.

    Input parameter:
        model: the siamese model which is for encoding
        whale_data: train data without new_whale
    return:  
        all_embedding_training (ndarray)
    '''
    if preCalculated:
        all_embedding_training = pickle.load(
            "./input/all_embedding_training.pl")
        return all_embedding_training
    else:
        all_embedding_training = []
        for im, label in whale_data:
            embedding_single_im = model.get_embedding(im)
            all_embedding_training.append(embedding_single_im)

        all_embedding_training = np.array(all_embedding_training)
        pickle.dump(all_embedding_training,
                    "./input/all_embedding_training.pl")
        return all_embedding_training


if __name__ == '__main__':
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    # the train data we used have 5004 whale ids
    num_classes = 5004

    model = SiameseNet(EmbeddingNet()).to(device)
    model.load_state_dict(torch.load(
        './trained_model/Siamese_7thTrain.model'))

    model.eval()

    # define a whale test data set, since the url path is written within the data set class
    TEST_PATH = "../Humpback-Whale-Identification-1st--master/input/test/"

    # ---- Transform operation from img_transform module----
    # Just use the basic one
    transform_img = img_transform.transforms_img()
    images_test = np.array(os.listdir(TEST_PATH))
    # print(len(images_test))

    # ---- load all the embedding of training data set----
    dst_train = get_train_data()

    id_index_dict = load_label_index_dict()  # dict: map whale_id -> index id
    indexId_data_train = [id_index_dict[label]
                          for label in dst_train['Id']]

    training_set_embedding = get_all_embedding_training(
        model, WhaleDatasetTrain(dst_train['Image'], indexId_data_train, transform_train=transform_img), preCalculated=False)

    # ---- Load the test data set, we can use the previous whale data set----
    dataset_test = WhaleDatasetTest(
        images_test, transform_img)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=32, shuffle=False, num_workers=1, pin_memory=True)

    index_id = np.array(dataset_test.id_list)  # index -> whale_id
    test_classnames = []
    softmax_output = torch.nn.Softmax()

    for test_batch in tqdm(dataloader_test):
        '''
        I think there will be a lot of bugs... 
        Since the in the second line return batch embedding.
        '''
        test_batch = test_batch.to(device, dtype=torch.float)
        embeddings_test = model.get_embedding(test_batch)

        score_with_all_train = [calculate_score(embeddings_test, single_train_embedding)
                                for single_train_embedding in training_set_embedding]
        score_data_frame_with_all_train = pd.DataFrame(
            {"score": score_with_all_train, "id": indexId_data_train})
        score_group = score_data_frame_with_all_train.groupby(
            ['id']).max().sort_value(by=['score'])['score']
        top_5_score_whale_id = np.array(
            [whale_id for whale_id in score_group.index[:5]])
        top_5_whale_labels = index_id[top_5_score_whale_id].tolist()

        score_group_cal_threshold = np.mean(
            [score_im for score_im in score_group])

        cal_threshold_base = softmax_output(torch.tensor(score_group))
        if cal_threshold_base[0] > 0.1:
            top_5_whale_labels = ['new_whale'] + top_5_whale_labels[:4]

        test_classnames.extend([" ".join(s) for s in whale_labels])

    testdf = pd.DataFrame({'Image': images_test, 'Id': test_classnames})
    testdf.to_csv(
        './results/submission_{}_siamese.csv'.format(model_name), index=False)

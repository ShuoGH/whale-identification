import torch.nn as nn
import torch.nn.functional as F
from whale_data_siamese import SiameseTrainData
# import img_transform
from PIL import Image
import torch
import torchvision


class EmbeddingNet(nn.Module):
    '''
    Base model to encode the images into embedding.
    This model is based on resnet18.

    The embedding is encoded by the resnet18.

    Reference:
        1. @adambielski https://github.com/adambielski/siamese-triplet Thanks to adam, it's a very useful repo.
    '''

    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.basemodel = torchvision.models.resnet18()
        self.basemodel.fc = nn.Linear(512, 256)

    def forward(self, x):
        output = self.basemodel(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


# class EmbeddingNetL2(EmbeddingNet):
#     '''
#     not sure what's this for
#     '''

#     def __init__(self):
#         super(EmbeddingNetL2, self).__init__()

#     def forward(self, x):
#         output = super(EmbeddingNetL2, self).forward(x)
#         output /= output.pow(2).sum(1, keepdim=True).sqrt()
#         return output

#     def get_embedding(self, x):
#         return self.forward(x)


class SiameseNet(nn.Module):
    '''
    You shouls input a base model to initialize this Siamesenet.
    '''

    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


if __name__ == '__main__':
    # def transform_train_img(img):
    #     '''
    #     input: cv2.imread image.
    #     return: transformed PIL from torchvision.transform
    #     '''
    #     # do a series of transform on images
    #     img_processed = img_transform.random_gaussian_noise(img, sigma=0.1)
    #     img_processed = img_transform.random_angle_rotate(img_processed)
    #     img_processed = img_transform.random_crop(img_processed)
    #     img = Image.fromarray(img_processed)

    #     transform_basic = img_transform.transforms_img()
    #     return transform_basic(img)

    # model = SiameseNet(EmbeddingNet())
    # # print(model)
    # loss_function = nn.CrossEntropyLoss()
    # # optimiser = optim.Adam(model.parameters())

    # dst = SiameseTrainData(transform_img=transform_train_img)
    # dst_train = torch.utils.data.DataLoader(
    #     dst, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    # # img_pair, id_flag = dst[6]
    # model.train()
    # for x_batch, y_batch in dst_train:
    #     print(x_batch[0][1].size())
    #     print(len(y_batch))
    #     # outputs = model(x_batch[0], x_batch[1])
    # print(outputs)
    print("If you want to test, edit this file.")

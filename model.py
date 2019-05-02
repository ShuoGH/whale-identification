import torchvision
import torch.nn as nn


class model_whale(nn.Module):
    '''
    parameter:
        num_classes: the output channels in the final fully connected layer
    '''

    def __init__(self, num_classes=5005, inchannels=3, model_name='resnet18'):
        super().__init__()
        self.model_name = model_name
        if self.model_name == 'resnet18':
            self.basemodel = torchvision.models.resnet18(pretrained=True)
            self.basemodel.fc = nn.Linear(512, num_classes)
        elif self.model_name == 'resnet50':
            self.basemodel = torchvision.models.resnet50(pretrained=True)
            self.basemodel.fc = nn.Linear(2048, num_classes)
        elif self.model_name == 'resnet101':
            self.basemodel = torchvision.models.resnet101(pretrained=True)
            self.basemodel.fc = nn.Linear(2048, num_classes)
        elif self.model_name == 'resnet152':
            self.basemodel = torchvision.models.resnet152(pretrained=True)
            self.basemodel.fc = nn.Linear(2048, num_classes)

    def freeze(self):
        '''
        Just freeze the pretrained parameter to realize the transfer learning.

        '''
        for p in self.basemodel.parameters():
            p.requires_grad = False  # Freeze all existing layers
        if self.model_name.find('resnet') > -1:
            for param in self.basemodel.layer4.parameters():
                param.requires_grad = True
            self.basemodel.fc.weight.requires_grad = True  # unfreeze last layer weights
            self.basemodel.fc.bias.requires_grad = True  # unfreeze last layer biases

    def forward(self, x):
        out = self.basemodel(x)
        return out

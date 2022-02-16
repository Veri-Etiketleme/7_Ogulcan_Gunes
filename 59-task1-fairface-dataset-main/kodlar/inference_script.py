from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torchvision.io import read_image
import os
import copy

cudnn.benchmark = True

from sklearn.preprocessing import LabelEncoder
label_encoder1 = LabelEncoder()
label_encoder2 = LabelEncoder()
label_encoder3 = LabelEncoder()

img_labels = pd.read_csv('../fairface-data/fairface_label_train.csv')

img_labels['age'] = label_encoder1.fit_transform(img_labels['age'])
img_labels['gender'] = label_encoder2.fit_transform(img_labels['gender'])
img_labels['race'] = label_encoder3.fit_transform(img_labels['race'])

def get_label_names(labels):
    '''
    gets true labels corresponding to the encoded values
    '''
    age, gender, race = labels
    age = label_encoder1.inverse_transform(age)
    gender = label_encoder2.inverse_transform(gender)
    race = label_encoder3.inverse_transform(race)
    return age, gender, race

def imshow(inp, title=None):
    '''
    for displaying a input image 
    '''
    inp = read_image(inp)
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    plt.show()

def test_image(img, model):
    '''
    for testing an image on the saved model
    and returning the output labels
    '''
    img = read_image(img).unsqueeze(0)
    img = img.float().cuda()
    op1, op2, op3 = model(img)
    op1 = torch.argmax(torch.softmax(op1, dim=1), dim=1).cpu()
    op2 = torch.argmax(torch.softmax(op2, dim=1), dim=1).cpu()
    op3 = torch.argmax(torch.softmax(op3, dim=1), dim=1).cpu()
    age, gender, race = get_label_names((op1, op2, op3))
    return age, gender, race


class MultiTaskClassifier(nn.Module):
    '''
    Building the architecture of the pre-trained backbone
    '''
    def __init__(self):
        '''
        defining the fully-connected layers added on top of the backbone
        '''
        super().__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.num_ftrs = self.resnet50.fc.out_features
        self.op1 = nn.Linear(self.num_ftrs, 9)
        self.op2 = nn.Linear(self.num_ftrs, 2)
        self.op3 = nn.Linear(self.num_ftrs, 7)

    def forward(self, x):
        '''
        for returning the label values predicted by the model
        '''
        x = self.resnet50(x)
        age = self.op1(x)
        gender = self.op2(x)
        race = self.op3(x)
        return (age, gender, race)

PATH = "models/resnet50_32.pth"
testmodel = MultiTaskClassifier().cuda()
testmodel.load_state_dict(torch.load(PATH))

if __name__ == "__main__":

    img_path = os.path.join("../fairface-data/", img_labels.iloc[0,0])
    imshow(img_path)
    print(test_image(img_path, testmodel))

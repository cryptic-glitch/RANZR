import torch
import torch.nn as nn
import sys

sys.path.append('../input/timmyy/pytorch-image-models-master')
import timm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import re
import copy
from torchvision.utils import save_image
import json
import glob
import torch.nn.functional as F
from skimage import color, io, transform
import torchvision
from torchvision import models, datasets, transforms

path = r'../input/teacher-11-label-complete/teacher_11_label_complete.csv'

img_path = r'../input/ranzcr-clip-catheter-line-classification/train'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Teacher_Annotation(Dataset):
    def __init__(self, csv_file, root_dir):
        self.da_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.da_frame)

    def __getitem__(self, idx):
        im = os.path.join(self.root_dir, self.da_frame.iloc[idx, 0] + '.jpg')
        im_name = self.da_frame.iloc[idx, 0]

        im_label = json.loads(self.da_frame.iloc[idx, 2])
        im_label = np.array(im_label)

        im_read = cv2.imread(im, 0)
        image_histt = cv2.createCLAHE(clipLimit=3)
        imagee = image_histt.apply(im_read)

        trans_img = transforms.Compose(
            [transforms.ToPILImage(), transforms.RandomResizedCrop((624, 624), scale=(0.85, 1.0)),
             transforms.RandomRotation((-8, 8)), transforms.Grayscale(num_output_channels=3),
             transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        return {'image': trans_img(imagee), 'im_label': im_label}


Teacher_dataset = Teacher_Annotation(csv_file=path, root_dir=img_path)
Teacher_Dataloader = DataLoader(Teacher_dataset, batch_size=6, shuffle=True)


def Rinz_Model(model, criterion, optimizer, num_epochs=30):
    for i in range(num_epochs):
        for inputs in Teacher_Dataloader:
            input = inputs['image'].to(device)
            labels = inputs['im_label'].to(device)
            optimizer.zero_grad()
            output = model(input)
            labels = labels.type_as(output)
            labels = labels.unsqueeze(1)
            output = output.unsqueeze(1)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        print(i)
        if i == 29:
            torch.save(model.state_dict(), 'Phase3_624_30epoch.pth')

    return model


class SeResNet152D(nn.Module):
    def __init__(self, model_name='seresnet152d'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True).to(device)

        for p in self.model.layer1.parameters():
            p.requires_grad = False
        for p in self.model.layer2.parameters():
            p.requires_grad = False

        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, 11).to(device)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return output


my_model = SeResNet152D()
teacher_path = r'../input/phase2-1515-624-thick2/Phase2_30_624.pth'
teacher = torch.load(teacher_path)
kits = list(teacher.keys())[:272]
pre_dic = {i: j for i, j in teacher.items() if i in kits}
my_model.load_state_dict(pre_dic, strict=False)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(my_model.parameters(), lr=0.001, momentum=0.9)
Rinz_Model(model=my_model, criterion=criterion, optimizer=optimizer, num_epochs=30)

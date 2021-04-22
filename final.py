import torch
import torch.nn as nn
import sys
sys.path.append(r'../input/timmyy/pytorch-image-models-master')
import timm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import re
import copy
from torchvision.utils import save_image
import json
import glob
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from skimage import color, io, transform
import torchvision
from torchvision import models, datasets, transforms
from torch.optim import lr_scheduler

path = r'../input/11-full-label/ultimate_11_label_teacher_deleted_concatenated.csv'

img_path = r'../input/ranzcr-clip-catheter-line-classification/train'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class student_dataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.da_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.da_frame)

    def __getitem__(self, idx):
        im = os.path.join(self.root_dir, self.da_frame.iloc[idx, 0] + '.jpg')
        im_name = self.da_frame.iloc[idx, 0]

        im_label = json.loads(self.da_frame.iloc[idx, 1])
        im_label = np.array(im_label)

        im_read = cv2.imread(im, 0)
        image_histt = cv2.createCLAHE(clipLimit=3)
        imagee = image_histt.apply(im_read)

        trans_img = transforms.Compose(
            [transforms.ToPILImage(), transforms.RandomResizedCrop((624, 624), scale=(0.85, 1.0)),
             transforms.RandomHorizontalFlip(p=0.2), transforms.RandomRotation((-7, 7)),
             transforms.Grayscale(num_output_channels=3),
             transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        return {'image': trans_img(imagee), 'im_label': im_label}


student_dataset = student_dataset(csv_file=path, root_dir=img_path)

y = []
for i in range(len(student_dataset)):
    y.append(list(student_dataset[i]['im_label']))
X_train, X_test, y_train, y_test = train_test_split(list(range(len(student_dataset))), y, test_size=0.2, random_state=42)
dataset = {}
dataset['train'] = Subset(student_dataset, X_train)
dataset['val'] = Subset(student_dataset, X_test)
dataloaders = {x: DataLoader(dataset[x], batch_size=6, shuffle=False) for x in ["train", "val"]}
dataloader_auc = {x: DataLoader(dataset[x], batch_size=1, shuffle=False) for x in ["train", "val"]}
def Rinz_Model(model, criterion, optimizer,scheduler, num_epochs=30):
    epoch_loss = []
    for i in range(num_epochs):
        running_loss = 0.0
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            for inputs in dataloaders[phase]:
                input_image = inputs['image'].to(device)
                labels = inputs['im_label'].to(device)
                optimizer.zero_grad()
                output = model(input_image)
                labels = labels.type_as(output)
                labels = labels.unsqueeze(1)
                output = output.unsqueeze(1)
                loss = criterion(output, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    running_loss += loss * input_image.size(0)

                if phase == 'val':
                    running_loss += loss * input_image.size(0)

            epoch_loss.append(running_loss / len(dataset[phase]))
        print(epoch_loss)
        scheduler.step()
        if i == 29:
            torch.save(model.state_dict(), 'final_seresnet152d_30_ADAM.pth')
            model.eval()
            for ph in ['train', 'val']:
                output = []
                true_label = []
                for ii in dataloader_auc[ph]:
                    ima = ii['image'].to(device)
                    output.append(model(ima).squeeze(0).sigmoid().detach().tolist())
                    true_label.append(ii['im_label'].squeeze(0).detach().to(dtype=torch.float32).tolist())

                print(roc_auc_score(true_label, output, average=None))
                print(roc_auc_score(true_label, output, average='macro'))

    return model


class SeResNet152D(nn.Module):
    def __init__(self, model_name='seresnet152d'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False).to(device)
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
teacher_path = r'../input/teacher-11-1515-624/phase3_30_624.pth'
my_model.load_state_dict(torch.load(teacher_path))

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(my_model.parameters(), lr=0.001)
exp_scheduler =  lr_scheduler.StepLR(optimizer,step_size =9,gamma = 0.1)
Rinz_Model(model=my_model, criterion=criterion, optimizer=optimizer, scheduler = exp_scheduler ,num_epochs=30)

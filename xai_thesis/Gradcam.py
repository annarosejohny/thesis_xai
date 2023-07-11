import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms as T

from tqdm import tqdm
import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import utils, csv

def rename_csv():
    # Open the original CSV file and the new CSV file
    with open(CSV_FILE, 'r') as original_file, open('modified.csv', 'w', newline='') as modified_file:
        # Create CSV reader and writer objects
        reader = csv.reader(original_file)
        
        writer = csv.writer(modified_file)
        
        # Iterate over each row in the original CSV file
        for row in reader:
        
            # Modify the row data by adding the desired extension
            modified_row = [row[0] + '_rot_aligned.jpg'] + row[1:]
            
            # Write the modified row to the new CSV file
            writer.writerow(modified_row)



class ImageModel(nn.Module):

    def __init__(self, num_classes):
        super(ImageModel, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4,4), stride=2),

            nn.Conv2d(in_channels=64, out_channels = 128, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4,4), stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4,4), stride=2),


            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), padding=1),
            nn.ReLU()
        )

        self.maxpool = nn.MaxPool2d(kernel_size=(4,4), stride=2)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12800, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        self.gradient = None

    def activations_hook(self, grad):
        self.gradient = grad

    def forward(self, images):

        x = self.feature_extractor(images) #activations map
        h = x.register_hook(self.activations_hook)
        x = self.maxpool(x)
        x = self.classifier(x)
        return x

    def get_activation_gradients(self):#a1, a2,...ak
        return self.gradient

    def get_activations(self,x): #A1, A2, ....AK
        return self.feature_extractor(x) #64*8*8

def train_fn(dataloader, model, optimizer, criterion):
    trainloader = dataloader
    DEVICE = 'cpu'
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(trainloader):
        # plt.imshow(images[0].permute(1,2,0))
        # plt.title(labels[0])
        # plt.show()
   
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
       
        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss/len(dataloader)


def eval_fn(trainloader, dataloader, model, criterion):
   
    DEVICE = 'cpu'
    model.eval()
    total_loss = 0.0
    for images, labels in tqdm(trainloader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item()

    return total_loss/len(dataloader)



if __name__ == "__main__":

    # CSV_FILE = '/Users/annarosejohny/Downloads/gmdb_train_images_v1.0.3.csv'
    DATA_DIR = '/Users/annarosejohny/Desktop/XAI_EYE/data/GestaltMatcherDB/v1.0.3/gmdb_align/'
    CSV_FILE = '//Users/annarosejohny/Desktop/XAI_EYE/data/GestaltMatcherDB/v1.0.3/modified.csv'

    DEVICE = 'cpu'
    BATCH_SIZE = 16
    LR = 0.001
    EPOCHS = 20

    data = pd.read_csv(CSV_FILE)
    train_df, valid_df = train_test_split(data, test_size=0.2, random_state=42)

    train_augs = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485,0.456, 0.406], std=[0.229, 0.224,0.225])
        ])

    valid_augs = A.Compose([
        A.Normalize(mean=[0.485,0.456, 0.406], std=[0.229, 0.224,0.225])
        ])
    
    trainset = utils.ImageDataset(train_df, augs = train_augs, data_dir=DATA_DIR)
    validset = utils.ImageDataset(valid_df, augs = valid_augs, data_dir=DATA_DIR)

    image, label = trainset[0]
    # print(f"No. of examples in the trainset {len(trainset)}")
    # print(f"No. of examples in the validset {len(validset)}")
    # plt.imshow(image.permute(1,2,0))
    # plt.title(label)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = DataLoader(validset, batch_size=BATCH_SIZE)
    for images, labels in trainloader:
        break

    # print(f"One batch image shape : {images.shape}")
    # print(f"One batch label shape : {labels.shape}")
    num_classes = 204
    model = ImageModel(num_classes).to(DEVICE)
    # model.to(DEVICE)


    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    best_valid_loss = np.Inf

    for epoch in range(EPOCHS):
        # total_loss = 0.0
        # for i, data in enumerate(trainloader,0):
        #     # plt.imshow(images[0].permute(1,2,0))
        #     # plt.title(labels[0])
        #     # plt.show()
   
        #     images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
       
        #     optimizer.zero_grad()
        #     logits = model(images)
        #     loss = criterion(logits, labels)
        #     loss.backward()
        #     optimizer.step()

        #     total_loss += loss.item()
        #     if i%100 == 99:
        #        print(f"Epoch [{EPOCHS+1}/{EPOCHS}], Step [{i+1}/{len(trainloader)}], Loss: {running_loss/100:.4f}")
        #        total_loss = 0.0

        # print("Training finished.")



        train_loss = train_fn(trainloader, model, optimizer, criterion)
        valid_loss = eval_fn(trainloader, validloader, model, criterion)
    
        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), 'best_weights.pt')
            best_valid_loss = valid_loss

            print("SAVED WEIGHTS SUCCESS")

        print(f"EPOCH : {i+ 1} TRAIN LOSS : {train_loss} VALID LOSS : {valid_loss}")


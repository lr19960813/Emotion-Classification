import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import torchvision.models.vit
import clip
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score
from sklearn import metrics
import pandas as pd


class Artemis_training(Dataset):

    def __init__(self):  # 载入数据
        csv = pd.read_csv('/home/lirui/Downloads/code/artemis_dataset/resnet/label_training2.csv', low_memory=False)
        model, preprocess = clip.load("ViT-B/32", device=0, jit=False)
        self.emotion = csv['emotion']
        self.style = csv['artstyle']
        self.repetition = csv['repetition']
        self.file_name = csv['painting']
        self._len = len(self.repetition)
        self.painting_root = '/home/lirui/Downloads/code/artemis_dataset/wikiart'
        self.preprocess = preprocess

    def __getitem__(self, index):  # 根据索引index返回相应位置的图像及标签
        image_path = os.path.join(self.painting_root, self.style[index], self.file_name[index] + '.jpg')
        image = Image.open(image_path)
        image = self.preprocess(image)
        repetition = self.repetition[index]
        emotion = self.emotion[index]
        return image, emotion, repetition

    def __len__(self):  # 查看数据长度
        return self._len


class Artemis_validation(Dataset):

    def __init__(self):  # 载入数据
        csv = pd.read_csv('/home/lirui/Downloads/code/artemis_dataset/resnet/label_validation2.csv', low_memory=False)
        model, preprocess = clip.load("ViT-B/32", device=0, jit=False)
        self.emotion = csv['emotion']
        self.style = csv['artstyle']
        self.repetition = csv['repetition']
        self.file_name = csv['painting']
        self._len = len(self.repetition)
        self.painting_root = '/home/lirui/Downloads/code/artemis_dataset/wikiart'
        self.preprocess = preprocess

    def __getitem__(self, index):  # 根据索引index返回相应位置的图像及标签
        image_path = os.path.join(self.painting_root, self.style[index], self.file_name[index] + '.jpg')
        image = Image.open(image_path)
        image = self.preprocess(image)
        repetition = self.repetition[index]
        emotion = self.emotion[index]
        return image, emotion, repetition

    def __len__(self):  # 查看数据长度
        return self._len


class Artemis_testing(Dataset):

    def __init__(self):  # 载入数据
        csv = pd.read_csv('/home/lirui/Downloads/code/artemis_dataset/resnet/label_testing2.csv', low_memory=False)
        model, preprocess = clip.load("ViT-B/32", device=0, jit=False)
        self.emotion = csv['emotion']
        self.style = csv['artstyle']
        self.repetition = csv['repetition']
        self.file_name = csv['painting']
        self._len = len(self.repetition)
        self.painting_root = '/home/lirui/Downloads/code/artemis_dataset/wikiart'
        self.preprocess = preprocess

    def __getitem__(self, index):  # 根据索引index返回相应位置的图像及标签
        image_path = os.path.join(self.painting_root, self.style[index], self.file_name[index] + '.jpg')
        image = Image.open(image_path)
        image = self.preprocess(image)
        repetition = self.repetition[index]
        emotion = self.emotion[index]
        return image, emotion, repetition

    def __len__(self):  # 查看数据长度
        return self._len



def main():
    batch_size = 16
    lr = 1e-5
    epochs = 30

    convertor_train = Artemis_training()  # 实例化
    convertor_validate = Artemis_validation()
    train_loader = DataLoader(convertor_train, batch_size=batch_size, drop_last=True,shuffle=True)
    validate_loader = DataLoader(convertor_validate,batch_size=batch_size,drop_last=False,shuffle=True)

    train_num = len(convertor_train)
    val_num = len(convertor_validate)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training

    checkpoint = torch.load("D:/Dataset/clip_model/exp_lr/clip5e-06_0.8_2.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16


    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    print("using {} images for training, {} images for validation.".format(train_num,val_num))

    # Loss function
    loss_function = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
    # learning rate scheduler
    # scheduler_exp = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)
    # lr_exp = []

    for epoch in range(epochs):
        # train
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            optimizer.zero_grad()
            images, labels, repetition = data
            images = images.cuda()
            tensor_image = model(images)

            logits = tensor_image
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels, val_repetition = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')




net = model.image_encoder

in_channel = nn.Linear.in_features
net.fc = nn.Linear(in_channel, 9) # classification amount
net.to(device)

